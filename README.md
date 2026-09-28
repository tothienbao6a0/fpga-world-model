# FPGA world-model planning prototype

This repo now contains a **synthesizable planning engine**, its bit-exact software reference, and a small closed-loop benchmark. It tests a concrete hardware question: can a fixed-point circuit evaluate candidate futures for a learned dynamics model and choose an action within a predictable cycle budget?

The first engine is intentionally narrow. It rolls out a learned two-state linear model using Q8.8 arithmetic, scores nine streamed four-step plans, and returns the first action of the lowest-cost plan. The toy simulator and model fitting live in `worldbench/`; the hardware, reference, and verification live in `fpga/`.

## Run it

Install Python 3, [Icarus Verilog](https://steveicarus.github.io/iverilog/), and [Yosys](https://yosyshq.net/yosys/). On macOS, `brew install icarus-verilog yosys` provides the two hardware tools. Then run:

```sh
make check
python3 -m fpga.bench
python3 -m fpga.bench --environment nonlinear
make synth-report
```

`make check` runs unit tests, compiles and simulates the RTL against 22 bit-exact cases, and checks that Yosys can synthesize the top module. `fpga.bench` compares the fixed-point hardware arithmetic contract with floating-point planning in the same toy control task. `make synth-report` reports generic logic cells, **not** resources or timing for a chosen FPGA.

## Current result

With the default 24 episodes and 16 decisions per episode, the fixed-point contract changed the locally chosen action on 4.43% of linear-environment decisions and 3.91% of nonlinear-environment decisions relative to floating-point planning. Each decision streams 36 actions after a start cycle, so the ideal uninterrupted engine schedule is 37 cycles. These are reproducible toy-model results and a cycle count from the interface contract. There is no board implementation, measured clock rate, end-to-end latency, power result, or comparison with a GPU yet.

Start with the [hardware interface and limits](docs/hardware/rollout-engine.md), the [research index](docs/README.md), and the [September 2026 evidence review](docs/research/2026-09-direction-review.md).
