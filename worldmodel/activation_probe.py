"""Drive the candidate QKV RTL tile with activations from the real predictor."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

from worldmodel.bench import load_predictor
from worldmodel.candidates import predict_candidates
from worldmodel.quant import quantize_candidate_tile, quantized_tile
from worldmodel.spec import MODEL
from worldmodel.verify_candidate_rtl import traffic_bytes, verify


def dequantized_outputs(
    integer_outputs: tuple[tuple[int, ...], ...],
    activation_scale: float,
    weight_scales: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value * activation_scale * scale for value, scale in zip(row, weight_scales))
                 for row in integer_outputs)


def error_stats(reference: tuple[tuple[float, ...], ...], actual: tuple[tuple[float, ...], ...]) -> dict:
    if not reference or len(reference) != len(actual) or any(len(a) != len(b) for a, b in zip(reference, actual)):
        raise ValueError("reference and actual outputs must have matching nonempty rows")
    differences = [a - b for reference_row, actual_row in zip(reference, actual)
                   for a, b in zip(reference_row, actual_row)]
    values = [value for row in reference for value in row]
    if not differences:
        raise ValueError("output rows must be nonempty")
    squared_error = sum(value * value for value in differences)
    squared_reference = sum(value * value for value in values)
    return {
        "max_abs_error": max(abs(value) for value in differences),
        "rmse": math.sqrt(squared_error / len(differences)),
        "relative_l2_error": math.sqrt(squared_error / squared_reference) if squared_reference else None,
        "values_compared": len(differences),
    }


def full_qkv_error_stats(inputs, reference, weights, biases, scale_scope: str) -> dict:
    """Evaluate INT8 QKV error over every candidate and visual token in software."""
    import torch

    if (inputs.ndim != 3 or reference.ndim != 3 or weights.ndim != 2 or biases.ndim != 1
            or inputs.shape[:2] != reference.shape[:2] or inputs.shape[2] != weights.shape[1]
            or reference.shape[2] != weights.shape[0] or biases.shape[0] != weights.shape[0]
            or min(inputs.shape) < 1):
        raise ValueError("QKV input, output, weight, and bias shapes must align")
    if scale_scope == "token":
        activation_scales = inputs.abs().amax(dim=(0, 2)) / 127
    elif scale_scope == "layer":
        activation_scales = inputs.abs().max().expand(inputs.shape[1]) / 127
    else:
        raise ValueError("scale scope must be token or layer")
    activation_scales = torch.where(activation_scales > 0, activation_scales, 1.0)
    weight_scales = weights.abs().amax(dim=1) / 127
    weight_scales = torch.where(weight_scales > 0, weight_scales, 1.0)
    quantized_inputs = (inputs / activation_scales[None, :, None]).round().clamp(-127, 127)
    quantized_weights = (weights / weight_scales[:, None]).round().clamp(-127, 127)
    quantized_biases = (biases[None, :] / (activation_scales[:, None] * weight_scales[None, :])).round()
    integer_outputs = quantized_inputs @ quantized_weights.T + quantized_biases[None, :, :]
    actual = integer_outputs * activation_scales[None, :, None] * weight_scales[None, None, :]
    differences = (actual - reference).flatten()
    reference_norm = reference.norm().item()
    return {
        "max_abs_error": differences.abs().max().item(),
        "p95_abs_error": torch.quantile(differences.abs(), 0.95).item(),
        "rmse": differences.square().mean().sqrt().item(),
        "relative_l2_error": differences.norm().item() / reference_norm if reference_norm else None,
        "values_compared": differences.numel(),
    }


def probe(
    upstream: Path, checkpoint: Path, *, frames: int = 2,
    candidates: int = 4, token_indices: tuple[int, ...] = (0, 255, 256, 511),
    device: str = "cpu", cpu_threads: int = 1,
) -> dict:
    if not 1 <= frames <= MODEL.configured_frames or candidates < 2 or cpu_threads < 1:
        raise ValueError("frames must be 1..4, candidates at least 2, and threads positive")
    if not token_indices or any(index < 0 or index >= frames * MODEL.grid_side**2 for index in token_indices):
        raise ValueError("token indices must fit the frame count")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    predictor, proprio_encoder, _, digest = load_predictor(upstream, checkpoint, device)
    qkv = predictor.predictor_blocks[0].attn.qkv
    captured_inputs = []
    captured_outputs = []

    def capture_input(_module, args):
        captured_inputs.append(args[0].detach().cpu())

    def capture_output(_module, _args, output):
        captured_outputs.append(output[:, :, :16].detach().cpu())

    before = qkv.register_forward_pre_hook(capture_input)
    after = qkv.register_forward_hook(capture_output)
    try:
        generator = torch.Generator(device="cpu").manual_seed(1907)
        latent = torch.randn(1, frames, 1, MODEL.grid_side, MODEL.grid_side, MODEL.latent_dim,
                             generator=generator).to(device)
        proprio = torch.randn(1, frames, MODEL.proprio_dim, generator=generator).to(device)
        actions = torch.randn(candidates, frames, MODEL.action_dim, generator=generator).to(device)
        with torch.inference_mode():
            encoded_proprio = proprio_encoder(proprio).expand(-1, -1, MODEL.grid_side**2, -1)
            predict_candidates(predictor, latent, actions, encoded_proprio, "batch")
    finally:
        before.remove()
        after.remove()
    if len(captured_inputs) != 1 or len(captured_outputs) != 1:
        raise AssertionError("expected exactly one block-0 QKV invocation")

    weights = tuple(tuple(row) for row in qkv.weight[:16].detach().cpu().tolist())
    biases = tuple(qkv.bias[:16].detach().cpu().tolist())
    full_errors = {
        scope: full_qkv_error_stats(
            captured_inputs[0], captured_outputs[0], qkv.weight[:16].detach().cpu(),
            qkv.bias[:16].detach().cpu(), scope,
        )
        for scope in ("token", "layer")
    }
    layer_activation_scale = captured_inputs[0].abs().max().item() / 127
    rows = []
    for token_index in token_indices:
        float_activations = tuple(tuple(row) for row in captured_inputs[0][:, token_index, :].tolist())
        reference = tuple(tuple(row) for row in captured_outputs[0][:, token_index, :].tolist())
        modes = {}
        for name, scale in (("token_scale", None), ("layer_scale", layer_activation_scale)):
            quantized_activations, quantized_weights, quantized_biases, activation_scale, weight_scales = (
                quantize_candidate_tile(float_activations, weights, biases, activation_scale=scale)
            )
            integer_outputs = tuple(quantized_tile(row, quantized_weights, quantized_biases)
                                    for row in quantized_activations)
            rtl_result = verify(quantized_activations, quantized_weights, quantized_biases)
            reconstructed = dequantized_outputs(integer_outputs, activation_scale, weight_scales)
            modes[name] = {
                "rtl_result": rtl_result,
                "activation_scale": activation_scale,
                **error_stats(reference, reconstructed),
            }
        rows.append({
            "token_index": token_index,
            "candidate_input_max_abs_delta": max(abs(a - b) for row in float_activations[1:]
                                                  for a, b in zip(float_activations[0], row)),
            "quantization_modes": modes,
        })
    return {
        "model": "facebook/jepa-wms:jepa_wm_pusht predictor",
        "upstream_commit": MODEL.upstream_commit,
        "checkpoint_sha256": digest,
        "input_kind": "synthetic_encoded_latents_actions_and_proprioception; actual_block_0_qkv_activations",
        "measurement_scope": "all tokens for first 16 QKV rows in software; sampled tokens in RTL; no full-model INT8 quality or FPGA latency",
        "device": device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cpu_threads": cpu_threads if device == "cpu" else None,
        "float_precision": str(latent.dtype),
        "tile_precision": "symmetric_int8_activations_and_per_row_weights_int32_accumulators",
        "frames": frames,
        "candidates": candidates,
        "tile_rows": 16,
        "tile_input_columns": MODEL.predictor_dim,
        "layer_activation_scale": layer_activation_scale,
        "traffic": traffic_bytes(16, MODEL.predictor_dim, candidates),
        "full_16_row_software_quantization": full_errors,
        "samples": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(".model-cache/jepa-wms"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(".model-cache/models--facebook--jepa-wms/snapshots")
        / MODEL.checkpoint_revision / MODEL.checkpoint_filename,
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[0, 255, 256, 511])
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = probe(
            args.upstream, args.checkpoint, frames=args.frames,
            candidates=args.candidates, token_indices=tuple(args.tokens),
            device=args.device, cpu_threads=args.cpu_threads,
        )
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
