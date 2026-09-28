# Streaming rollout engine

The first hardware implementation is a synchronous, synthesizable SystemVerilog module at `fpga/rollout_engine.sv`. It takes a two-state learned dynamics model and evaluates action sequences streamed by a host. This proves an arithmetic and control interface, not a board-level accelerator.

## Host contract

All state, targets, and weights use signed Q8.8 in 16 bits. Each 64-bit weight bus packs four signed weights in the order `[position, velocity, action, bias]`, least-significant 16 bits first. The weights predict the next position and velocity independently. Valid actions are signed two-bit `-1`, `0`, and `1`.

After reset, pulse `start` for one rising edge with the initial state, target, and both weight buses stable. The engine captures these values and raises `action_ready`. Stream exactly `NUM_CANDIDATES × HORIZON` actions in candidate order, one per cycle when `action_valid && action_ready`. Bubbles are allowed. `start` is ignored while busy. After the last accepted action, `done` pulses for one cycle; `best_action` and `best_cost` hold the winning first action and unsigned Q16 cost until the next start. Ties keep the earliest streamed candidate. Parameters must be positive; the tested default is nine candidates and four actions per candidate.

Each predicted state uses a four-term Q8.8 dot product. The sum is shifted right by eight bits and saturated to signed 16 bits. The score is the final position error squared, plus `(9830 × final velocity squared) >> 16`, plus 2621 Q16 units for each nonzero action. The software oracle in `fpga/reference.py` exactly mirrors the integer operations. The older `worldbench` quantizer rounds intermediate operations differently, so it is a separate research baseline.

The engine processes one action per accepted cycle. With no bubbles, the default command takes one start cycle and 36 action cycles to assert `done`. This excludes host transfer, clock setup, and any board-level interface. The module contains no AXI, DMA, memory controller, or clock constraint.

## Reproduce the current checks

Run `make check` from the repo root. `fpga/verify.py` compiles the RTL with Icarus Verilog, streams 22 cases, and compares both output fields against the Python oracle. Cases include learned weights from linear and nonlinear training sets, a tie, Q8.8 saturation, and bubbles in the action stream. Yosys checks generic synthesis and structural correctness. Run `make synth-report` to inspect the generic cell breakdown.

Run `python3 -m fpga.bench` and `python3 -m fpga.bench --environment nonlinear` to compare closed-loop decisions with the floating-point planner. The default nine-plan bank covers all first actions but is much smaller than the exhaustive 81-plan bank in the original four-step software baseline. Both planners in this comparison use the same nine plans.

The local tools checked for this prototype were Icarus Verilog 13.0 and Yosys 0.69. Their official upstream projects publish releases and maintain active source repositories: [Icarus Verilog](https://github.com/steveicarus/iverilog) and [Yosys](https://github.com/YosysHQ/yosys). Both are established Homebrew packages; no Python package dependency was added.

## What the next hardware step needs

Pick a target board and memory/host link, then run a vendor mapping and timing build. A real workload also needs a checkpoint-backed model, candidate generation, and a transfer-aware comparison with a CPU or GPU baseline. Generic Yosys cells and the ideal stream-cycle count cannot establish speed, energy, or FPGA resource use for a particular device.
