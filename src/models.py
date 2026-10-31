"""Shared Hugging Face model resolution: reuse local caches, download only when asked."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def model_path(repo: str, revision: str, *, filename: str | None = None, download: bool = False) -> str:
    """Reuse the default/project HF cache; download into whichever drive has space."""
    from huggingface_hub import hf_hub_download, snapshot_download

    default = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    roots = list(dict.fromkeys([default, Path.home() / ".cache/huggingface/hub",
                               PROJECT_ROOT / "models/hf-cache/hub"]))

    def fetch(root: Path, local: bool) -> str:
        kwargs = dict(repo_id=repo, revision=revision, cache_dir=str(root), local_files_only=local)
        if filename:
            return hf_hub_download(filename=filename, **kwargs)
        path = snapshot_download(
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.model", "*.jinja"],
            max_workers=2, **kwargs)
        if not any(Path(path).glob("*.safetensors")) and not any(Path(path).glob("*.bin")):
            raise FileNotFoundError(f"Weights missing from {path}")
        return path

    for root in roots:
        try:
            return fetch(root, True)
        except (OSError, ValueError):
            pass
    if not download:
        raise FileNotFoundError(f"Missing {repo}. Run scripts/infer.py --prepare-only first.")
    root = default
    probe = root
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < 4 * 1024**3:
        root = PROJECT_ROOT / "models/hf-cache/hub"
        if shutil.disk_usage(PROJECT_ROOT).free < 4 * 1024**3:
            raise OSError("Insufficient space in default cache and project drive")
    logger.info("Downloading %s to %s", repo, root)
    return fetch(root, False)
