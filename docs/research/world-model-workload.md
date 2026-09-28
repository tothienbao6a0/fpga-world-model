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

Next work: profile operator time and memory traffic on a GPU; obtain real encoded Push-T trajectories and ground-truth future latents; measure prediction quality before quantization; then implement and validate a single chosen operator in the F2 shell. The [official dataset](https://huggingface.co/datasets/facebook/jepa-wms) requires accepting access conditions, so no task data was downloaded in this run. This repo does not currently include a bitstream or cloud FPGA result.

The isolated runtime pins PyTorch 2.7.0 and matching torchvision 0.22.0 to stay close to the upstream project's declared `torch>=2.7.0` and exact torchvision requirement. [PyTorch's release history](https://github.com/pytorch/pytorch/releases) shows continued maintenance and [its package page](https://pypi.org/project/torch/) shows broad platform distribution. The upstream model imports `timm.layers`; [timm 1.0.30](https://github.com/huggingface/pytorch-image-models/releases) was a recent maintained release when checked. The model's own official repo demonstrates adoption of these libraries in this exact workload. `huggingface-hub` is pinned for the tested checkpoint download API. No upstream source or checkpoint is vendored here.
