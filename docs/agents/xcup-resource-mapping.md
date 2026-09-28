# Agent work: UltraScale+ tile resource mapping

## What ran

I checked the local hardware toolchain after confirming this Mac has no CUDA or MPS backend. The installed Yosys supports `synth_xilinx -family xcup`, so I mapped the existing single- and four-candidate QKV tiles to Xilinx UltraScale+ primitives. The question was how much resource cost accompanies four-way weight broadcast.

## What it changed

- `worldmodel/resource_report.py` runs both out-of-context mappings and extracts comparable DSP, LUT, carry, and flip-flop counts.
- `worldmodel/resource_report_test.py` checks the report parser against duplicate stat sections and missing output.
- `results/qkv_xcup_yosys_resource_mapping.json` records the tool version, options, counts, and ratios.
- `Makefile` runs the mapping in `make check`; the README and workload study state its limits. The RTL itself did not change.

## What it was not allowed to touch

No AWS instance, vendor toolchain, bitstream, or existing checkpoint was modified. This run cannot establish an F2 resource budget or timing result because it lacks the F2 shell and part-specific implementation flow.

## What a reviewer should check by hand

- **DSP cost scales with parallel candidates.** The mapped single tile uses 16 DSP48E2 cells and the four-candidate tile uses 64; the generated report and direct Yosys logs agreed.
- **LUT and register costs also scale near fourfold.** Counts were 1,039 versus 4,111 LUTs and 546 versus 2,082 FDRE cells. These are primitive counts before placement, not board utilization percentages.
- **No throughput follows from this mapping.** The design still requires F2 integration, memory interfaces, place and route, and a GPU comparison before any speedup claim.

`make check` and `make model-check` passed after this addition.
