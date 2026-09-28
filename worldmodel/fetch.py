"""Download the pinned JEPA-WM checkpoint into an ignored local cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldmodel.checkpoint import verify_checkpoint
from worldmodel.spec import MODEL


def fetch(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=MODEL.checkpoint_repo,
        revision=MODEL.checkpoint_revision,
        filename=MODEL.checkpoint_filename,
        cache_dir=str(cache_dir),
    ))
    verify_checkpoint(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".model-cache"))
    args = parser.parse_args(argv)
    print(fetch(args.cache_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
