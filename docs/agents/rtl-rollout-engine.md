# Agent work: first hardware rollout engine

## What ran

Codex turned the toy planning study into a synthesizable streaming engine after the user asked to build the actual project. It read the existing benchmark, repo context, and research notes, then checked the RTL with Icarus Verilog and generic Yosys synthesis. `make check` passed 22 Python tests, 22 bit-exact RTL scenarios, and the synthesis check. The default linear and nonlinear closed-loop benchmark commands completed with 384 decisions each.

## What it changed

- `fpga/rollout_engine.sv` implements Q8.8 learned-state rollout, cost accumulation, and best-action selection.
- `fpga/reference.py` provides the integer arithmetic oracle; `fpga/verify.py` drives the simulator and compares exact outputs.
- `fpga/bench.py` compares fixed-point and floating-point decisions in the existing toy task.
- `Makefile` supplies the reproducible test and synthesis gate; the hardware doc specifies the stream contract.

The boundary is a host-driven arithmetic core and toy benchmark. The prior software study remains available.

## What it was not allowed to touch

No board, host link, or pretrained checkpoint was specified, so this change does not claim mapped FPGA performance or hardware energy. It did not deploy to a device or publish a release.

## What a reviewer should check by hand

- **The 37-cycle figure is an ideal interface schedule.** It is one start edge plus 36 accepted action edges; action bubbles or host transfer add cycles. Verified from the RTL handshake and the 9×4 candidate bank, not a device timing report.
- **The simulation proves arithmetic agreement only for tested streams.** The 22 scenarios cover fitted models, ties, saturation, and bubbles. They do not prove every input or every parameterization. Check `fpga/verify.py` before extending the engine's protocol.
- **The closed-loop result depends on a small candidate bank.** Both compared planners use the same nine sequences; the original software study used 81 exhaustive length-four sequences. Verified from `fpga/bench.py` and `worldbench/planner.py`.
