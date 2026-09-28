"""Checkpoint file verification shared by fetching and inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from worldmodel.spec import MODEL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != MODEL.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 does not match the pinned official release")
    return actual
