"""Map QKV tiles to UltraScale+ primitives and record unplaced resource counts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TILES = {
    "single_candidate": ("rtl/qkv_tile.sv", "qkv_tile"),
    "four_candidates": ("rtl/qkv_candidate_tile.sv", "qkv_candidate_tile"),
}
CELL_TYPES = ("DSP48E2", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "CARRY4", "FDRE")


def parse_resources(log: str, top: str) -> dict[str, int]:
    """Read the final local cell counts from Yosys's stat section."""
    marker = f"=== {top} ==="
    offset = log.rfind(marker)
    if offset < 0:
        raise ValueError(f"Yosys output has no final {top} statistics")
    section = log[offset:]
    counts = {}
    for cell in CELL_TYPES:
        match = re.search(rf"^\s*(\d+)\s+{cell}\s*$", section, flags=re.MULTILINE)
        counts[cell] = int(match.group(1)) if match else 0
    if not counts["DSP48E2"] or not sum(counts[f"LUT{size}"] for size in range(2, 7)):
        raise ValueError(f"Yosys did not map expected DSP and LUT cells for {top}")
    counts["LUT_total"] = sum(counts[f"LUT{size}"] for size in range(2, 7))
    return counts


def map_tile(path: Path, top: str) -> dict[str, int]:
    relative_path = path.relative_to(ROOT).as_posix()
    command = (
        f"read_verilog -sv {relative_path}; "
        f"synth_xilinx -family xcup -top {top} -noiopad -noclkbuf; stat"
    )
    result = subprocess.run(["yosys", "-Q", "-T", "-p", command],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    return parse_resources(result.stdout, top)


def report() -> dict:
    version = subprocess.run(["yosys", "-V"], check=True, capture_output=True, text=True).stdout.strip()
    tiles = {name: map_tile(ROOT / path, top) for name, (path, top) in TILES.items()}
    return {
        "tool": version,
        "mapping": "Yosys synth_xilinx -family xcup -noiopad -noclkbuf",
        "scope": "out-of-context, unplaced primitive mapping; no F2 shell, routing, timing, or power",
        "parameters": {"lanes": 16, "inputs": 400, "single_candidates": 1, "broadcast_candidates": 4},
        "tiles": tiles,
        "broadcast_to_single_ratio": {
            cell: tiles["four_candidates"][cell] / tiles["single_candidate"][cell]
            for cell in ("DSP48E2", "LUT_total", "CARRY4", "FDRE")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(report(), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
