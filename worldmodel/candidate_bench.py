"""Measure exact reuse across candidate actions for the official Push-T predictor."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

from worldmodel.bench import load_predictor, sync_device
from worldmodel.candidates import predict_candidates
from worldmodel.spec import MODEL, workload_macs


def measure_candidates(
    upstream: Path, checkpoint: Path, *, device: str = "cpu", frames: int = 2,
    counts: tuple[int, ...] = (1, 2, 4, 8), warmup: int = 1,
    repeats: int = 5, cpu_threads: int = 1,
) -> dict:
    if not counts or min(counts) < 1 or warmup < 0 or repeats < 1 or cpu_threads < 1:
        raise ValueError("counts, repeats, and cpu_threads must be positive; warmup may be zero")
    workload_macs(frames)
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    predictor, proprio_encoder, weights, digest = load_predictor(upstream, checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(1907)
    latent = torch.randn(1, frames, 1, MODEL.grid_side, MODEL.grid_side, MODEL.latent_dim, generator=generator).to(device)
    proprio = torch.randn(1, frames, MODEL.proprio_dim, generator=generator).to(device)
    all_actions = torch.randn(max(counts), frames, MODEL.action_dim, generator=generator).to(device)
    rows = []
    with torch.inference_mode():
        encoded_proprio = proprio_encoder(proprio).expand(-1, -1, MODEL.grid_side**2, -1)
        for count in counts:
            actions = all_actions[:count]
            reference = predict_candidates(predictor, latent, actions, encoded_proprio, "serial")
            timings = {}
            errors = {}
            for mode in ("serial", "batch", "shared_input"):
                for _ in range(warmup):
                    predict_candidates(predictor, latent, actions, encoded_proprio, mode)
                sync_device(torch, device)
                samples = []
                for _ in range(repeats):
                    start = perf_counter()
                    output = predict_candidates(predictor, latent, actions, encoded_proprio, mode)
                    sync_device(torch, device)
                    samples.append((perf_counter() - start) * 1000)
                error = (output - reference).abs().max().item()
                # BLAS changes reduction order when candidates are batched.
                if not torch.allclose(output, reference, rtol=1e-4, atol=2e-4):
                    raise AssertionError(f"{mode} differs from serial official predictor at {count} candidates")
                timings[mode] = {"p50_ms": statistics.median(samples), "min_ms": min(samples)}
                errors[mode] = error
            serial_ms = timings["serial"]["p50_ms"]
            rows.append({
                "candidates": count,
                "major_macs_no_reuse": workload_macs(frames, count)["total_major_macs"],
                "timings": timings,
                "max_abs_error_vs_serial": errors,
                "batch_speedup_vs_serial": serial_ms / timings["batch"]["p50_ms"],
                "shared_input_speedup_vs_serial": serial_ms / timings["shared_input"]["p50_ms"],
            })
    return {
        "model": "facebook/jepa-wms:jepa_wm_pusht predictor",
        "upstream_commit": MODEL.upstream_commit,
        "checkpoint_sha256": digest,
        "input_kind": "synthetic_encoded_latents_actions_and_proprioception",
        "measurement_scope": "predictor_only; no visual encoder, checkpoint load, or host transfer",
        "device": device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cpu_threads": cpu_threads if device == "cpu" else None,
        "precision": str(latent.dtype),
        "frames": frames,
        "latent_shape": list(latent.shape),
        "predictor_weight_bytes": sum(t.numel() * t.element_size() for t in weights.values()),
        "warmup": warmup,
        "repeats": repeats,
        "rows": rows,
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
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = measure_candidates(
            args.upstream, args.checkpoint, device=args.device, frames=args.frames,
            counts=tuple(args.counts), warmup=args.warmup,
            repeats=args.repeats, cpu_threads=args.cpu_threads,
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
