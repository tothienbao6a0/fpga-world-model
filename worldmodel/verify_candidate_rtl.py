"""Verify broadcast QKV hardware with synthetic and checkpoint weight slices."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from worldmodel.quant import quantized_tile
from worldmodel.verify_rtl import checkpoint_case, hex_packed, synthetic_case

RTL = Path(__file__).resolve().parent.parent / "rtl/qkv_candidate_tile.sv"


def candidate_vectors(base: tuple[int, ...], count: int) -> tuple[tuple[int, ...], ...]:
    """Distinct, deterministic INT8 vectors sharing the reference scale."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    return tuple(tuple(max(-127, min(127, value + (candidate * (index % 5 - 2))))
                       for index, value in enumerate(base)) for candidate in range(count))


def testbench(
    activations: tuple[tuple[int, ...], ...],
    weights: tuple[tuple[int, ...], ...],
    biases: tuple[int, ...],
) -> str:
    if not activations or not weights or len(weights) != len(biases) or not weights[0]:
        raise ValueError("tile dimensions must be nonempty and aligned")
    candidates, lanes, inputs = len(activations), len(weights), len(weights[0])
    if any(len(row) != inputs for row in activations + weights):
        raise ValueError("all candidates and weights must have the same input width")
    expected = tuple(value for row in activations for value in quantized_tile(row, weights, biases))
    lines = [
        "`timescale 1ns/1ps", "module tb;",
        "reg clk=0; always #5 clk=~clk;",
        "reg rst_n=0, start=0, input_valid=0;",
        f"reg [{lanes*32-1}:0] biases=0;",
        f"reg [{candidates*8-1}:0] activations=0;",
        f"reg [{lanes*8-1}:0] weights=0;",
        "wire input_ready, done;",
        f"wire [{candidates*lanes*32-1}:0] results;",
        f"qkv_candidate_tile #(.LANES({lanes}), .INPUTS({inputs}), .CANDIDATES({candidates})) dut (",
        " .clk(clk), .rst_n(rst_n), .start(start), .biases(biases),",
        " .input_valid(input_valid), .activations(activations), .weights(weights),",
        " .input_ready(input_ready), .done(done), .results(results));",
        "initial begin",
        " repeat (2) @(negedge clk); rst_n=1;",
        " @(negedge clk);",
        f" biases={lanes*32}'h{hex_packed(biases, 32)}; start=1;",
        " @(negedge clk); start=0;",
        " if (!input_ready) $fatal(1, \"tile did not start\");",
    ]
    for index in range(inputs):
        if index == inputs // 2:
            lines.extend([" input_valid=0; @(negedge clk);", " if (!input_ready) $fatal(1, \"tile lost ready\");"])
        column = tuple(row[index] for row in weights)
        activation_column = tuple(row[index] for row in activations)
        lines.extend([
            f" input_valid=1; activations={candidates*8}'h{hex_packed(activation_column, 8)};",
            f" weights={lanes*8}'h{hex_packed(column, 8)}; @(negedge clk);",
        ])
    lines.extend([
        " input_valid=0; #1;",
        f" if (!done || results !== {candidates*lanes*32}'h{hex_packed(expected, 32)})",
        "   $fatal(1, \"candidate QKV mismatch: done=%0d results=%h\", done, results);",
        f' $display("PASS {candidates} candidates x {lanes} lanes x {inputs} inputs");',
        " $finish;", "end", "endmodule",
    ])
    return "\n".join(lines) + "\n"


def verify(
    activations: tuple[tuple[int, ...], ...],
    weights: tuple[tuple[int, ...], ...],
    biases: tuple[int, ...],
) -> str:
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        raise RuntimeError("Icarus Verilog is required for RTL verification")
    with TemporaryDirectory() as directory:
        tb = Path(directory) / "tb.sv"
        executable = Path(directory) / "tb.vvp"
        tb.write_text(testbench(activations, weights, biases), encoding="utf-8")
        subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(RTL), str(tb)],
                       check=True, capture_output=True, text=True)
        result = subprocess.run(["vvp", str(executable)], check=True, capture_output=True, text=True)
        return result.stdout.splitlines()[0]


def traffic_bytes(lanes: int, inputs: int, candidates: int) -> dict[str, int]:
    if min(lanes, inputs, candidates) < 1:
        raise ValueError("tile dimensions must be positive")
    return {
        "serial_weight_bytes": lanes * inputs * candidates,
        "broadcast_weight_bytes": lanes * inputs,
        "activation_bytes_both": inputs * candidates,
        "macs_both": lanes * inputs * candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args(argv)
    base, weights, biases = synthetic_case()
    print(verify(candidate_vectors(base, args.candidates), weights, biases))
    if args.checkpoint:
        (base, weights, biases), _ = checkpoint_case(args.checkpoint)
        print(verify(candidate_vectors(base, args.candidates), weights, biases))
        print(json.dumps(traffic_bytes(len(weights), len(base), args.candidates)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
