"""Pinned upstream model and arithmetic shape for the Push-T predictor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    upstream_url: str = "https://github.com/facebookresearch/jepa-wms.git"
    upstream_commit: str = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
    checkpoint_repo: str = "facebook/jepa-wms"
    checkpoint_revision: str = "bb2d9cf0ee9060f83103b134d7c52e82bf7e2a47"
    checkpoint_filename: str = "jepa_wm_pusht.pth.tar"
    checkpoint_sha256: str = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
    grid_side: int = 16
    latent_dim: int = 384
    predictor_dim: int = 400
    action_dim: int = 10
    proprio_dim: int = 4
    proprio_embedding_dim: int = 16
    blocks: int = 6
    heads: int = 16
    configured_frames: int = 4


MODEL = ModelSpec()


def workload_macs(frames: int, batch: int = 1, spec: ModelSpec = MODEL) -> dict[str, int]:
    """Approximate dense MACs from the checkpoint shapes; excludes norms and RoPE."""
    if frames < 1 or frames > spec.configured_frames or batch < 1:
        raise ValueError("frames must be 1..4 and batch must be positive")
    tokens = frames * spec.grid_side**2
    width = spec.predictor_dim
    matrix_per_block = 12 * tokens * width**2  # QKV, projection, two MLP layers
    attention_per_block = 2 * tokens**2 * width  # QK^T and attention-value
    return {
        "tokens_per_sample": tokens,
        "matrix_macs": batch * spec.blocks * matrix_per_block,
        "attention_macs": batch * spec.blocks * attention_per_block,
        "total_major_macs": batch * spec.blocks * (matrix_per_block + attention_per_block),
    }


def validate_weight_shapes(weights: dict[str, tuple[int, ...]], spec: ModelSpec = MODEL) -> None:
    """Reject checkpoints whose predictor does not match the pinned architecture."""
    expected = {
        "predictor_embed.weight": (spec.latent_dim, spec.latent_dim),
        "action_encoder.weight": (spec.predictor_dim, spec.action_dim),
        "predictor_blocks.0.attn.qkv.weight": (3 * spec.predictor_dim, spec.predictor_dim),
        "predictor_blocks.0.mlp.fc1.weight": (4 * spec.predictor_dim, spec.predictor_dim),
        "predictor_blocks.0.adaLN_modulation.1.weight": (6 * spec.predictor_dim, spec.predictor_dim),
        f"predictor_blocks.{spec.blocks - 1}.attn.qkv.weight": (3 * spec.predictor_dim, spec.predictor_dim),
        "predictor_proj.weight": (spec.latent_dim, spec.latent_dim),
    }
    for name, shape in expected.items():
        if weights.get(name) != shape:
            raise ValueError(f"checkpoint {name} has shape {weights.get(name)}, expected {shape}")
    if any(name.startswith(f"predictor_blocks.{spec.blocks}.") for name in weights):
        raise ValueError("checkpoint has more predictor blocks than the pinned architecture")
