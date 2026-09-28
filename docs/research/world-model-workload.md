# Checkpoint-backed world-model inference workload

## Target and scope

The target is the released `jepa_wm_pusht` checkpoint from [Meta's JEPA-WMs repository](https://github.com/facebookresearch/jepa-wms). Its [model card and checkpoint](https://huggingface.co/facebook/jepa-wms) are public. The source is pinned to commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`; the checkpoint is pinned to revision `bb2d9cf0ee9060f83103b134d7c52e82bf7e2a47` and SHA256 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`.

The checkpoint contains a six-block `VisionTransformerAdaLN` predictor and a proprioception encoder. Its predictor has 17,626,480 parameters and 70,505,920 bytes of FP32 weights. Each frame contributes a 16×16 grid of 384-dimensional visual tokens. The action input has 10 values per frame; raw proprioception has four values and is embedded into 16 feature channels. The predictor's working width is 400 across 16 attention heads. These dimensions were verified against the downloaded checkpoint and a strict weight load into the official upstream class. The DINO visual encoder is a separate pretrained component and is **not** included in these first timings.

## First measured baseline

The benchmark uses seeded synthetic encoded latents, actions, and proprioception. It loads the trained predictor and proprioception weights, executes the official predictor class, and times 20 forward passes after three warmups. The device is a single-threaded macOS ARM CPU with PyTorch 2.7.0. GPU and FPGA results have not been measured.

| Frames | Visual tokens | Approx. major MACs | Median predictor time | p95 predictor time |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 7.16 billion | 98.9 ms | 111.2 ms |
| 4 | 1,024 | 16.83 billion | 295.4 ms | 301.5 ms |

The operation estimate counts QKV and output projections, two MLP layers, and QK/attention-value products. It excludes layer norms, RoPE, AdaLN, input/output projections, and memory movement. It is an analytical estimate from the real tensor shapes, not a hardware performance measurement. The measured outputs change when actions are zeroed, confirming the loaded predictor is action-conditioned for these synthetic inputs; that is not evidence of physical prediction quality.

## Hardware question

The first candidate hardware problem is **reuse of predictor weights and visual tokens across the six attention/MLP blocks** under the AWS F2 memory hierarchy. At four frames, attention pairwise work grows faster than the matrix work as the context length rises. A serious FPGA claim must compare the mapped block and full host path against optimized CPU/GPU execution at matched shape and precision, report resource use and timing closure, and verify numerical error against the original predictor. The hardware architecture is not chosen by this CPU timing alone.

## First hardware block

`rtl/qkv_tile.sv` computes 16 rows of an INT8 matrix-vector product while streaming one activation and 16 weights per accepted cycle. It takes 400 accepted cycles for a full input channel pass, plus a start cycle, and returns 16 signed INT32 accumulators with quantized bias. The real QKV weight matrix in block 0 has shape 1,200×400; the 16-lane tile covers one of 75 row groups. `make check` simulates a small synthetic tile and synthesizes the default 16×400 module with Yosys. `make model-check` also quantizes the first 16 rows of the released QKV weights and checks all 16 RTL outputs against the integer oracle. The test reports maximum absolute dequantization error against FP32 for those rows and a seeded synthetic activation vector.

This is a **functional tile**, not an F2 integration or a competitive mapping. It streams weights from the host every cycle, so it does not solve the weight-traffic problem identified above. Full-layer latency, DSP use, placement, timing, host I/O, and prediction quality under INT8 remain unmeasured. The tile gives a correctness baseline for a later weight-stationary or batched design.

## Candidate-action reuse

One visual context and proprioceptive state can be evaluated against many candidate action sequences. In this checkpoint, the actions pass through AdaLN modulation before attention and MLP in **every** predictor block. Thus a full block output cannot be cached exactly across candidates. The visual input projection and proprioception embedding are action independent; `shared_input` computes the former once and broadcasts it. `batch` uses the official predictor on a candidate batch, and `serial` invokes it once per action. The three paths are checked against serial output with FP32 tolerance; the largest observed absolute difference was 0.000107, attributable to different reduction orders when batched. This is numerical agreement on synthetic encoded inputs, not task-quality validation.

The [recorded run](../../results/jepa_wm_pusht_candidate_cpu_macos_arm64.json) uses the pinned checkpoint and source, two frames, one macOS ARM CPU thread, FP32, five timed repeats after one warmup, with predictor-only timing and no host transfer or visual encoder. Median milliseconds were:

| Candidates | Serial | Official batch | Shared input projection |
| ---: | ---: | ---: | ---: |
| 1 | 76.4 | 76.0 | 77.0 |
| 2 | 150.1 | 148.2 | 145.1 |
| 4 | 301.2 | 288.6 | 291.4 |
| 8 | 604.0 | 587.4 | 577.7 |

The eight-candidate CPU gain from exact input sharing was 4.6%, with only five samples; it is not evidence of a robust GPU or FPGA gain. The dominant action-conditioned blocks still run eight times. The QKV broadcast tile is a separate hardware experiment: four synthetic quantized candidate activation vectors consume the same 16 streamed checkpoint weights each cycle, with 64 INT8 MACs in parallel. The vectors use one shared quantization scale. On a 16-row × 400-column slice of block 0's real checkpoint QKV matrix, simulation matched the integer oracle for all four candidates. Four serial passes stream 25,600 weight bytes; broadcast streams 6,400 weight bytes. Both perform 25,600 MACs, and broadcast requires four times the multiplier lanes and fourfold activation bandwidth. These are tile interface counts, not measured DRAM traffic or FPGA latency. Full QKV, attention, MLP, timing closure, resource utilization, and prediction quality remain open.

## Checkpoint activation probe

The [activation probe result](../../results/jepa_wm_pusht_qkv_candidate_activations_cpu_macos_arm64.json) advances the tile input from arbitrary vectors to **actual block-0 QKV activations** captured while the official checkpoint evaluates four different actions against one synthetic encoded visual context. Candidate QKV inputs differ (sampled maximum absolute differences from candidate 0: 0.72–1.30). For the first 16 of 1,200 QKV output rows, software INT8 quantization was compared against the official FP32 QKV outputs at all 512 tokens and four candidates, or 32,768 scalar outputs:

| Activation scale | Relative L2 error | p95 absolute error | Max absolute error |
| --- | ---: | ---: | ---: |
| One scale per token, shared across candidates | 1.96% | 0.127 | 0.297 |
| One scale for the layer, shared across tokens and candidates | 2.38% | 0.154 | 0.360 |

Weights use a symmetric scale per output row; biases are quantized into the corresponding INT32 accumulator scale. For four sampled tokens (0, 255, 256, 511), both scaling modes passed RTL simulation against the integer oracle for all four candidates and 16 output rows. The layer scale is simpler to supply to hardware but had higher error in this one synthetic-context probe. The full 32,768-output error study is software quantization; RTL checked 512 scalar outputs across the four sampled tokens and two scaling modes. These measurements do not establish full-model accuracy, task quality on Push-T, or FPGA timing.

Next work: profile operator time and memory traffic on a GPU; obtain real encoded Push-T trajectories and ground-truth future latents; measure prediction quality before quantization; then implement and validate a single chosen operator in the F2 shell. The [official dataset](https://huggingface.co/datasets/facebook/jepa-wms) requires accepting access conditions, so no task data was downloaded in this run. This repo does not currently include a bitstream or cloud FPGA result.

The isolated runtime pins PyTorch 2.7.0 and matching torchvision 0.22.0 to stay close to the upstream project's declared `torch>=2.7.0` and exact torchvision requirement. [PyTorch's release history](https://github.com/pytorch/pytorch/releases) shows continued maintenance and [its package page](https://pypi.org/project/torch/) shows broad platform distribution. The upstream model imports `timm.layers`; [timm 1.0.30](https://github.com/huggingface/pytorch-image-models/releases) was a recent maintained release when checked. The model's own official repo demonstrates adoption of these libraries in this exact workload. `huggingface-hub` is pinned for the tested checkpoint download API. No upstream source or checkpoint is vendored here.
