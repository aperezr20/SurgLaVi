import logging
import os
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MODELS = {
    "SurgCLIP-B": {
        "hf_repo": "aleperez24/SurgCLIP",
        "hf_filename": "surgclip_beta.pth",
        "url": "https://huggingface.co/aleperez24/SurgCLIP/resolve/main/surgclip_beta.pth",
        "sha256": None,
    },
}


def available_models() -> list:
    return list(_MODELS.keys())


def get_cache_dir() -> Path:
    custom = os.environ.get("SURGCLIP_CACHE", None)
    root = Path(custom) if custom else (Path.home() / ".cache" / "surgclip")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _verify_sha256(path: Path, expected: str) -> None:
    import hashlib
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    if sha.hexdigest() != expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}. Delete it and retry.")


def _try_hf_download(hf_repo: str, hf_filename: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub not installed, falling back to direct URL.")
        return False
    shutil.copy(hf_hub_download(repo_id=hf_repo, filename=hf_filename), dest)
    return True


def download_weights(model_name: str, force_download: bool = False) -> Path:
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {available_models()}")

    entry = _MODELS[model_name]
    dest = get_cache_dir() / f"{model_name}.pth"

    if dest.exists() and not force_download:
        return dest
    if dest.exists():
        dest.unlink()

    if not _try_hf_download(entry["hf_repo"], entry["hf_filename"], dest):
        torch.hub.download_url_to_file(entry["url"], str(dest), progress=True)

    if entry.get("sha256"):
        _verify_sha256(dest, entry["sha256"])

    return dest