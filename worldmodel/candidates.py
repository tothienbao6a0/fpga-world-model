"""Exact candidate-action execution paths for the pinned JEPA-WM predictor."""

from __future__ import annotations


def predict_candidates(predictor, latent, actions, encoded_proprio, mode: str):
    """Return one prediction per action sequence from a shared visual context.

    latent: [1,T,1,H,W,384]; actions: [C,T,10]; encoded_proprio:
    [1,T,H*W,16]. All tensors must already be on the predictor device.
    """
    candidates, frames, _ = actions.shape
    if latent.shape[0] != 1 or encoded_proprio.shape[0] != 1:
        raise ValueError("a single shared visual/proprioceptive context is required")
    if candidates < 1 or frames != latent.shape[1] or frames != encoded_proprio.shape[1]:
        raise ValueError("candidate actions must match the context frame count")
    if mode == "serial":
        import torch

        return torch.cat(
            [predictor(latent, actions[i:i + 1], encoded_proprio)[0] for i in range(candidates)]
        )
    if mode == "batch":
        return predictor(
            latent.expand(candidates, -1, -1, -1, -1, -1),
            actions,
            encoded_proprio.expand(candidates, -1, -1, -1),
        )[0]
    if mode != "shared_input":
        raise ValueError(f"unknown candidate execution mode: {mode}")
    if predictor.proprio_encoding != "feature" or predictor.cond_tokens != 0 or predictor.proprio_emb_dim < 1:
        raise ValueError("shared_input requires the pinned feature-proprioception model")

    # The visual input projection is action independent. AdaLN makes every
    # subsequent transformer block action dependent, so none is cached here.
    visual = predictor.predictor_embed(latent).flatten(2, 4)
    patches = visual.shape[2]
    x = predictor.concat_obs(
        visual.expand(candidates, -1, -1, -1),
        encoded_proprio.expand(candidates, -1, -1, -1),
    ).flatten(1, 2)
    conditioning = predictor.action_encoder(actions)
    attention_mask = (
        predictor.attn_mask[:x.shape[1], :x.shape[1]].to(x.device, non_blocking=True)
        if predictor.attn_mask is not None else None
    )
    for block in predictor.predictor_blocks:
        x = block(
            x, conditioning, mask=None, attn_mask=attention_mask,
            T=frames, H_patches=predictor.grid_height,
            W_patches=predictor.grid_width, cond_tokens=predictor.cond_tokens,
        )
    x = predictor.predictor_norm(x)
    x = x.view(candidates, frames, patches, predictor.predictor_total_embed_dim)
    return predictor.predictor_proj(x[..., :-predictor.proprio_emb_dim])
