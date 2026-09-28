"""Run the official JEPA-WM Push-T predictor on checkpoint-backed synthetic latents."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from worldmodel.checkpoint import verify_checkpoint
from worldmodel.spec import MODEL, validate_weight_shapes, workload_macs


def verify_source(upstream: Path) -> None:
    if not (upstream / "app/plan_common/models/AdaLN_vit.py").is_file():
        raise ValueError("upstream path does not contain the JEPA-WM predictor")
    result = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip() != MODEL.upstream_commit:
        raise ValueError(f"upstream commit is {result.stdout.strip()}, expected {MODEL.upstream_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    )
    if dirty.stdout.strip():
        raise ValueError("upstream checkout has local modifications")


def sync_device(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_predictor(upstream: Path, checkpoint: Path, device: str):
    """Verify and load the pinned official predictor and proprioception encoder."""
    verify_source(upstream)
    checkpoint_sha = verify_checkpoint(checkpoint)
    import torch

    sys.path.insert(0, str(upstream))
    from app.plan_common.models.AdaLN_vit import VisionTransformerAdaLN
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    predictor_weights = {name.removeprefix("module."): value for name, value in payload["predictor"].items()}
    validate_weight_shapes({name: tuple(value.shape) for name, value in predictor_weights.items()})
    predictor = VisionTransformerAdaLN(
        img_size=224, patch_size=14, num_frames=MODEL.configured_frames, tubelet_size=1,
        embed_dim=MODEL.latent_dim, predictor_embed_dim=MODEL.latent_dim,
        depth=MODEL.blocks, num_heads=MODEL.heads, use_rope=True,
        local_window=(3, -1, -1), action_dim=MODEL.action_dim,
        proprio_dim=MODEL.proprio_dim, proprio_emb_dim=MODEL.proprio_embedding_dim,
        proprio_tokens=0, proprio_encoder_inpred=False, action_encoder_inpred=True,
    )
    predictor.load_state_dict(predictor_weights, strict=True)
    predictor = predictor.to(device).eval()
    proprio_encoder = ProprioceptiveEmbedding(
        num_frames=MODEL.configured_frames, tubelet_size=1,
        tokens_per_step=1, in_chans=MODEL.proprio_dim,
        embed_dim=MODEL.proprio_embedding_dim, shift_input=False,
    )
    proprio_weights = {name.removeprefix("module."): value for name, value in payload["proprio_encoder"].items()}
    proprio_encoder.load_state_dict(proprio_weights, strict=True)
    proprio_encoder = proprio_encoder.to(device).eval()

    return predictor, proprio_encoder, predictor_weights, checkpoint_sha


def run_benchmark(
    upstream: Path,
    checkpoint: Path,
    *,
    device: str = "cpu",
    batch: int = 1,
    frames: int = 2,
    warmup: int = 3,
    repeats: int = 20,
    cpu_threads: int = 1,
) -> dict:
    if batch < 1 or warmup < 0 or repeats < 1 or cpu_threads < 1:
        raise ValueError("batch, repeats, and cpu_threads must be positive; warmup may be zero")
    work = workload_macs(frames, batch)
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    predictor, proprio_encoder, predictor_weights, checkpoint_sha = load_predictor(upstream, checkpoint, device)

    generator = torch.Generator(device="cpu").manual_seed(1907)
    latent = torch.randn(batch, frames, 1, MODEL.grid_side, MODEL.grid_side, MODEL.latent_dim, generator=generator).to(device)
    actions = torch.randn(batch, frames, MODEL.action_dim, generator=generator).to(device)
    proprio = torch.randn(batch, frames, MODEL.proprio_dim, generator=generator).to(device)
    with torch.inference_mode():
        encoded_proprio = proprio_encoder(proprio).expand(-1, -1, MODEL.grid_side**2, -1)
        def predict(current_actions):
            return predictor(latent, current_actions, encoded_proprio)[0]

        for _ in range(warmup):
            predict(actions)
        sync_device(torch, device)
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            output = predict(actions)
            sync_device(torch, device)
            samples.append((perf_counter() - start) * 1000)
        zero_action_output = predict(torch.zeros_like(actions))
        sync_device(torch, device)
        action_effect = torch.linalg.vector_norm(output - zero_action_output).item()
        output_norm = torch.linalg.vector_norm(output).item()

    samples.sort()
    predictor_bytes = sum(t.numel() * t.element_size() for t in predictor_weights.values())
    return {
        "model": "facebook/jepa-wms:jepa_wm_pusht predictor",
        "upstream_commit": MODEL.upstream_commit,
        "checkpoint_sha256": checkpoint_sha,
        "input_kind": "synthetic_encoded_latents_and_actions",
        "device": device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cpu_threads": cpu_threads if device == "cpu" else None,
        "batch": batch,
        "frames": frames,
        "latent_shape": list(latent.shape),
        "output_shape": list(output.shape),
        "predictor_parameters": sum(t.numel() for t in predictor_weights.values()),
        "predictor_weight_bytes_fp32": predictor_bytes,
        "workload_estimate": work,
        "warmup": warmup,
        "repeats": repeats,
        "p50_predictor_ms": statistics.median(samples),
        "p95_predictor_ms": samples[min(len(samples) - 1, int(0.95 * (len(samples) - 1)))],
        "output_l2": output_norm,
        "action_effect_l2": action_effect,
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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(
            args.upstream, args.checkpoint, device=args.device, batch=args.batch,
            frames=args.frames, warmup=args.warmup, repeats=args.repeats,
            cpu_threads=args.cpu_threads,
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
