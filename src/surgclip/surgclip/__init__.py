import logging
from typing import Tuple, Union

import torch

from .config import get_config
from .download import available_models, download_weights
from .model import SurgCLIP
from .tokenizer import load_tokenizer, tokenize
from .transforms import get_preprocess, VideoPreprocessor

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["load", "tokenize", "available_models", "VideoPreprocessor", "get_preprocess"]


def load(
    model_name: str = "SurgCLIP-B",
    device: Union[str, torch.device] = None,
    num_frames: int = 16,
    temporal_modeling: bool = True,
    force_download: bool = False,
) -> Tuple[SurgCLIP, object, object]:
    """
    Load a pretrained SurgCLIP model. Downloads weights automatically on first call.

    Returns:
        (model, preprocess, tokenizer)
        - model:      SurgCLIP in eval() mode
        - preprocess: torchvision transform for a single PIL image -> (C, H, W) tensor
        - tokenizer:  BertTokenizer, pass to surgclip.tokenize()

    Example:
        model, preprocess, tokenizer = surgclip.load("SurgCLIP-B", device="cuda")
        tokens = surgclip.tokenize(["clipping", "dissection"], tokenizer, device="cuda")
        logits, _ = model(video, tokens)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    config = get_config(model_name, overrides={
        "device": str(device),
        "num_frames": num_frames,
        "model": {"temporal_modeling": {"enabled": temporal_modeling}},
    })

    tokenizer = load_tokenizer(config)
    model = SurgCLIP(config=config, tokenizer=tokenizer, is_pretrain=False)

    weights_path = download_weights(model_name, force_download=force_download)
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
    except Exception:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device).eval()
    preprocess = get_preprocess(config.inputs.image_res)
    return model, preprocess, tokenizer