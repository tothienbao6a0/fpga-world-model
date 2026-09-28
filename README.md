# World-model inference on reconfigurable hardware

This repository studies **the inference workload of a trained visual world model** and how to map its predictor to FPGA hardware. The first target is Meta's released JEPA-WM Push-T checkpoint. Given encoded visual tokens, actions, and proprioception, its six-block action-conditioned transformer predicts future visual latents. The work now starts with the real predictor and measured operator shapes; the earlier two-state planning toy is preserved under `archive/planning-prototype/` for provenance and is not the project baseline.

## What works now

- `worldmodel/bench.py` loads the official predictor and proprioception weights from a SHA-verified checkpoint into the official upstream implementation. It measures predictor latency with synthetic encoded visual tokens and actions, checks that actions affect the output, and reports tensor dimensions, parameter bytes, and major matrix/attention operation counts.
- `worldmodel/spec.py` pins the upstream commit and checkpoint revision and rejects a mismatched architecture.
- `rtl/qkv_tile.sv` is a synthesizable 16-lane INT8 matrix-vector tile for the real predictor's 1,200×400 QKV projection. The simulator checks a 16×400 slice quantized from the checkpoint against an integer software reference; the generic synthesis check runs in `make check`.
- `results/` contains the first one-thread macOS CPU measurements for two and four input frames. These are real **checkpoint-backed predictor** runs. The inputs are synthetic latents, so the numbers are not end-to-end video inference or prediction-quality results.

## Reproduce

Python 3.10, [uv](https://docs.astral.sh/uv/), Icarus Verilog, and Yosys are required for the complete checks. On macOS, install the hardware tools with `brew install icarus-verilog yosys`. The isolated model environment uses the pinned versions in `requirements-model.txt`. From the repo root:

```sh
make check
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-model.txt
mkdir -p .model-cache
git clone https://github.com/facebookresearch/jepa-wms.git .model-cache/jepa-wms
git -C .model-cache/jepa-wms checkout 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
.venv/bin/python -m worldmodel.fetch
.venv/bin/python -m worldmodel.bench --device cpu --frames 2
.venv/bin/python -m worldmodel.bench --device cpu --frames 4
make model-check
```

The checkpoint is about 212 MB and remains in the ignored local cache. The source checkout also stays ignored. The benchmark validates both pins, the source tree's clean state, and the checkpoint hash before loading weights. `make model-check` additionally compares a quantized QKV slice from that checkpoint with the RTL tile. The [official model weights](https://huggingface.co/facebook/jepa-wms) and [upstream code](https://github.com/facebookresearch/jepa-wms) use CC BY-NC 4.0; this repo redistributes neither.

## Next hardware milestone

The current predictor has 17.6 million parameters and takes 512 visual tokens for two frames or 1,024 for four frames. The measured four-frame predictor takes about 295 ms median on one Mac CPU thread; that is a comparison point, not an FPGA speedup claim. The next step is to profile its attention, MLP, and weight traffic on a suitable GPU and map one operator block through the AWS F2 toolchain, including host transfer and timing closure. See the [workload analysis](docs/research/world-model-workload.md) and [research index](docs/README.md).
