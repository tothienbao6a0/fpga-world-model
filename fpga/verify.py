"""Compile RTL and compare streamed rollouts with the bit-exact Python oracle."""

from __future__ import annotations

from random import Random
from shutil import which
from subprocess import run
from tempfile import TemporaryDirectory
from pathlib import Path

from fpga.reference import evaluate, model_weights, pack_weights, to_q
from worldbench.dynamics import fit_model, nonlinear_step, training_samples
from worldbench.planner import candidate_sequences

HORIZON = 4
SPARSE_CANDIDATES = 9
EXHAUSTIVE_CANDIDATES = 81


def literal(value: int, width: int) -> str:
    return f"- {width}'sd{-value}" if value < 0 else f"{width}'sd{value}"


def cases():
    rng = Random(1907)
    all_sequences = candidate_sequences(HORIZON)
    for index in range(20):
        dynamics = nonlinear_step if index % 2 else None
        model = fit_model(training_samples(256, index + 1, dynamics)) if dynamics else fit_model(training_samples(256, index + 1))
        weights_p, weights_v = model_weights(model)
        position = to_q(rng.uniform(-1.5, 1.5))
        velocity = to_q(rng.uniform(-0.5, 0.5))
        target = to_q(rng.choice((-1.25, 1.25)))
        candidates = tuple(rng.sample(all_sequences, SPARSE_CANDIDATES))
        yield index, position, velocity, target, weights_p, weights_v, candidates
    # Tie behavior and Q8.8 saturation are easy to miss with small random states.
    tie_candidates = tuple(((-1 if index % 2 == 0 else 1), 0, 0, 0) for index in range(SPARSE_CANDIDATES))
    yield 20, 0, 0, 0, (0, 0, 0, 0), (0, 0, 0, 0), tie_candidates
    yield 21, 32767, -32768, -32768, (32767, -32768, 32767, 0), (-32768, 32767, 0, 0), tuple(all_sequences[-SPARSE_CANDIDATES:])


def exhaustive_case():
    model = fit_model(training_samples(256, 41))
    weights_p, weights_v = model_weights(model)
    return (22, to_q(0.7), to_q(-0.3), to_q(1.25), weights_p, weights_v, candidate_sequences(HORIZON))


def testbench(case_rows, num_candidates: int) -> str:
    lines = [
        "`timescale 1ns/1ps",
        "module tb;",
        "reg clk = 0; always #5 clk = ~clk;",
        "reg rst_n = 0, start = 0, action_valid = 0;",
        "reg signed [15:0] initial_position = 0, initial_velocity = 0, target_position = 0;",
        "reg [63:0] position_weights = 0, velocity_weights = 0;",
        "reg signed [1:0] action = 0;",
        "wire action_ready, done;",
        "wire signed [1:0] best_action;",
        "wire [63:0] best_cost;",
        f"rollout_engine #(.HORIZON({HORIZON}), .NUM_CANDIDATES({num_candidates})) dut (",
        "  .clk(clk), .rst_n(rst_n), .start(start),",
        "  .initial_position(initial_position), .initial_velocity(initial_velocity),",
        "  .target_position(target_position), .position_weights(position_weights),",
        "  .velocity_weights(velocity_weights), .action_valid(action_valid), .action(action),",
        "  .action_ready(action_ready), .done(done), .best_action(best_action), .best_cost(best_cost));",
        "initial begin",
        "  repeat (2) @(negedge clk); rst_n = 1;",
    ]
    count = 0
    for index, position, velocity, target, weights_p, weights_v, candidates in case_rows:
        if len(candidates) != num_candidates or any(len(candidate) != HORIZON for candidate in candidates):
            raise ValueError("case shape does not match RTL parameters")
        expected = evaluate(position, velocity, target, weights_p, weights_v, candidates)
        lines.extend([
            "  @(negedge clk);",
            f"  initial_position = {literal(position, 16)};",
            f"  initial_velocity = {literal(velocity, 16)};",
            f"  target_position = {literal(target, 16)};",
            f"  position_weights = 64'h{pack_weights(weights_p):016x};",
            f"  velocity_weights = 64'h{pack_weights(weights_v):016x};",
            "  start = 1;",
            "  @(negedge clk); start = 0;",
            "  if (!action_ready) $fatal(1, \"engine did not start\");",
        ])
        for candidate in candidates:
            lines.append("  action_valid = 0; @(negedge clk);")
            for value in candidate:
                lines.append(f"  action_valid = 1; action = {literal(value, 2)}; @(negedge clk);")
        lines.extend([
            "  action_valid = 0; #1;",
            f"  if (!done || best_action !== {literal(expected.action, 2)} || best_cost !== 64'd{expected.cost})",
            f"    $fatal(1, \"case {index} failed: action=%0d cost=%0d done=%0d\", best_action, best_cost, done);",
        ])
        count += 1
    lines.extend([f'  $display("PASS {count} bit-exact rollout cases");', "  $finish;", "end", "endmodule"])
    return "\n".join(lines) + "\n"


def main() -> int:
    if not which("iverilog") or not which("vvp"):
        raise SystemExit("Icarus Verilog is required: install icarus-verilog, then rerun")
    source = Path(__file__).with_name("rollout_engine.sv")
    for num_candidates, case_rows in ((SPARSE_CANDIDATES, tuple(cases())), (EXHAUSTIVE_CANDIDATES, (exhaustive_case(),))):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tb = root / "tb.sv"
            output = root / "sim.vvp"
            tb.write_text(testbench(case_rows, num_candidates), encoding="utf-8")
            compile_result = run(["iverilog", "-g2012", "-s", "tb", "-o", str(output), str(source), str(tb)], capture_output=True, text=True)
            if compile_result.returncode:
                raise SystemExit(compile_result.stderr)
            simulation = run(["vvp", str(output)], capture_output=True, text=True)
            if simulation.returncode:
                raise SystemExit(simulation.stdout + simulation.stderr)
            print(f"{num_candidates} candidates: {simulation.stdout.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
