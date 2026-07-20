"""Download the VIVOS corpus from Kaggle into a destination dir (idempotent).

Standalone (no project imports) so it runs as `python -m src.vivos_download <dest>`
inside the container bootstrap. Auth uses a Kaggle token mounted at ~/.kaggle
(access_token or kaggle.json) or KAGGLE_USERNAME/KAGGLE_KEY env vars.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger("vivos_download")

_SLUG = "kynthesis/vivos-vietnamese-speech-corpus-for-asr"


def _find_root(base: Path) -> Path | None:
    """Dir that holds <split>/prompts.txt (i.e. parent of the split dir)."""
    for pf in Path(base).rglob("prompts.txt"):
        return pf.parent.parent
    return None


def ensure_vivos(dest: str | Path) -> Path:
    dest = Path(dest)
    existing = _find_root(dest) if dest.exists() else None
    if existing:
        log.info("VIVOS already present at %s", existing)
        return dest

    log.info("Downloading VIVOS from Kaggle (%s) ...", _SLUG)
    try:
        import kagglehub
        src = _find_root(Path(kagglehub.dataset_download(_SLUG)))
    except Exception as e:
        raise SystemExit(
            f"VIVOS download failed ({e}).\n"
            "Provide a Kaggle token (mount ~/.kaggle or set KAGGLE_USERNAME/KAGGLE_KEY), "
            f"or place the corpus manually so that {dest}/train/prompts.txt exists.\n"
            f"Dataset: https://www.kaggle.com/datasets/{_SLUG}"
        )
    if not src:
        raise SystemExit("Downloaded VIVOS but found no prompts.txt")

    dest.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():                       # copy train/ test/ (+ any extras)
        target = dest / child.name
        if child.is_dir() and not target.exists():
            log.info("copying %s -> %s", child.name, target)
            shutil.copytree(child, target)
    log.info("VIVOS ready at %s", dest)
    return dest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(ensure_vivos(sys.argv[1] if len(sys.argv) > 1 else "vivos"))
