# FPGA world-model planning prototype

This repo now contains a **synthesizable planning engine**, its bit-exact software reference, and a small closed-loop benchmark. It tests a concrete hardware question: can a fixed-point circuit evaluate candidate futures for a learned dynamics model and choose an action within a predictable cycle budget?

The first engine is intentionally narrow. It rolls out a learned two-state linear model using Q8.8 arithmetic, scores nine streamed four-step plans, and returns the first action of the lowest-cost plan. The toy simulator and model fitting live in `worldbench/`; the hardware, reference, and verification live in `fpga/`.

## Run it

Install Python 3, [Icarus Verilog](https://steveicarus.github.io/iverilog/), and [Yosys](https://yosyshq.net/yosys/). On macOS, `brew install icarus-verilog yosys` provides the two hardware tools. Then run:

```sh
make check
python3 -m fpga.bench
python3 -m fpga.bench --candidate-bank exhaustive
python3 -m fpga.sweep
make synth-report
```

`make check` runs unit tests, compiles and simulates the RTL against 23 bit-exact cases, including an 81-plan stream, and checks that Yosys can synthesize the top module. `fpga.bench` compares the fixed-point hardware arithmetic contract with floating-point planning in the same toy control task. `fpga.sweep` repeats both candidate banks and environments across five seeds. `make synth-report` reports generic logic cells, **not** resources or timing for a chosen FPGA.

## Current result

With the default nine-plan bank, each decision streams 36 actions after a start cycle, so the ideal uninterrupted engine schedule is 37 cycles. The exhaustive 81-plan bank takes 325 cycles. Across five toy-workload seeds, exhaustive search reduced local fixed-point action disagreements with the floating-point planner but increased mean final task cost in both environments. See the [candidate-breadth experiment](docs/experiments/003-candidate-breadth.md) for exact results. There is no board implementation, measured clock rate, end-to-end latency, power result, or comparison with a GPU yet.

Start with the [hardware interface and limits](docs/hardware/rollout-engine.md), the [research index](docs/README.md), and the [September 2026 evidence review](docs/research/2026-09-direction-review.md).
