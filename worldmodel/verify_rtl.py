"""Check an INT8 QKV tile against synthetic and released-checkpoint weights."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from worldmodel.checkpoint import verify_checkpoint
from worldmodel.quant import quantized_tile, symmetric_int8

RTL = Path(__file__).resolve().parent.parent / "rtl/qkv_tile.sv"


def hex_packed(values: tuple[int, ...], width: int) -> str:
    return "".join(f"{value & ((1 << width) - 1):0{width // 4}x}" for value in reversed(values))


def testbench(activations: tuple[int, ...], weights: tuple[tuple[int, ...], ...], biases: tuple[int, ...]) -> str:
    lanes, inputs = len(weights), len(activations)
    expected = quantized_tile(activations, weights, biases)
    lines = [
        "`timescale 1ns/1ps", "module tb;",
        "reg clk=0; always #5 clk=~clk;",
        "reg rst_n=0, start=0, input_valid=0;",
        f"reg [{lanes*32-1}:0] biases=0;",
        "reg signed [7:0] activation=0;",
        f"reg [{lanes*8-1}:0] weights=0;",
        "wire input_ready, done;",
        f"wire [{lanes*32-1}:0] results;",
        f"qkv_tile #(.LANES({lanes}), .INPUTS({inputs})) dut (",
        " .clk(clk), .rst_n(rst_n), .start(start), .biases(biases),",
        " .input_valid(input_valid), .activation(activation), .weights(weights),",
        " .input_ready(input_ready), .done(done), .results(results));",
        "initial begin",
        " repeat (2) @(negedge clk); rst_n=1;",
        " @(negedge clk);",
        f" biases={lanes*32}'h{hex_packed(biases, 32)}; start=1;",
        " @(negedge clk); start=0;",
        " if (!input_ready) $fatal(1, \"tile did not start\");",
    ]
    for index, activation in enumerate(activations):
        column = tuple(row[index] for row in weights)
        if index == inputs // 2:
            lines.extend([" input_valid=0; @(negedge clk);", " if (!input_ready) $fatal(1, \"tile lost ready during bubble\");"])
        lines.extend([
            f" input_valid=1; activation=8'h{activation & 255:02x};",
            f" weights={lanes*8}'h{hex_packed(column, 8)}; @(negedge clk);",
        ])
    lines.extend([
        " input_valid=0; #1;",
        f" if (!done || results !== {lanes*32}'h{hex_packed(expected, 32)})",
        "   $fatal(1, \"QKV tile mismatch: done=%0d results=%h\", done, results);",
        f' $display("PASS {lanes}x{inputs} INT8 QKV tile");',
        " $finish;", "end", "endmodule",
    ])
    return "\n".join(lines) + "\n"


def verify(activations: tuple[int, ...], weights: tuple[tuple[int, ...], ...], biases: tuple[int, ...]) -> str:
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        raise RuntimeError("Icarus Verilog is required for RTL verification")
    with TemporaryDirectory() as directory:
        tb = Path(directory) / "tb.sv"
        executable = Path(directory) / "tb.vvp"
        tb.write_text(testbench(activations, weights, biases), encoding="utf-8")
        subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(RTL), str(tb)], check=True, capture_output=True, text=True)
        result = subprocess.run(["vvp", str(executable)], check=True, capture_output=True, text=True)
        return result.stdout.splitlines()[0]


def synthetic_case():
    rng = random.Random(1907)
    activations = tuple(rng.randint(-127, 127) for _ in range(8))
    weights = tuple(tuple(rng.randint(-127, 127) for _ in activations) for _ in range(4))
    biases = (3, -7, 0, 127)
    return activations, weights, biases


def max_abs_dequantized_error(
    original_activations: tuple[float, ...],
    float_weights: list[list[float]],
    float_biases: list[float],
    integer_outputs: tuple[int, ...],
    activation_scale: float,
    weight_scales: tuple[float, ...],
) -> float:
    reconstructed = tuple(value * activation_scale * scale
                          for value, scale in zip(integer_outputs, weight_scales))
    reference = tuple(bias + sum(a * weight for a, weight in zip(original_activations, row))
                      for bias, row in zip(float_biases, float_weights))
    return max(abs(a - b) for a, b in zip(reconstructed, reference))


def checkpoint_case(checkpoint: Path):
    verify_checkpoint(checkpoint)
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["predictor"]
    qkv = state["module.predictor_blocks.0.attn.qkv.weight"][:16].tolist()
    bias = state["module.predictor_blocks.0.attn.qkv.bias"][:16].tolist()
    rng = random.Random(1907)
    original_activations = tuple(rng.gauss(0, 1) for _ in range(400))
    activations, activation_scale = symmetric_int8(original_activations)
    rows = [symmetric_int8(tuple(row)) for row in qkv]
    weights = tuple(row for row, _ in rows)
    biases = tuple(round(value / (activation_scale * scale)) for value, (_, scale) in zip(bias, rows))
    expected = quantized_tile(activations, weights, biases)
    max_abs_error = max_abs_dequantized_error(
        original_activations, qkv, bias, expected, activation_scale,
        tuple(scale for _, scale in rows),
    )
    return (activations, weights, biases), max_abs_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    print(verify(*synthetic_case()))
    if args.checkpoint:
        case, error = checkpoint_case(args.checkpoint)
        print(verify(*case))
        print(json.dumps({"layer": "predictor_blocks.0.attn.qkv", "rows_checked": 16,
                          "input_columns": 400, "max_abs_quantization_error": error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
