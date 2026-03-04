import copy
import json
from pathlib import Path
from types import SimpleNamespace

_CONFIGS_DIR = Path(__file__).parent / "configs"


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def _deep_update(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


_MODEL_DEFAULTS = {
    "SurgCLIP-B": dict(
        device="cpu",
        num_frames=16,
        num_workers=4,
        gradient_checkpointing=False,
        inputs=dict(
            image_res=224,
            batch_size_test=dict(video=4, image=16),
            video_input=dict(sample_type_test="middle", num_frames_test=16),
        ),
        model=dict(
            embed_dim=256,
            temp=0.07,
            vit_add_ln=True,
            vision_encoder=dict(
                name="vit_base_patch16_224",
                pretrained=True,
                d_model=768,
                config=str(_CONFIGS_DIR / "config_vit.json"),
            ),
            text_encoder=dict(
                name="bert_base",
                pretrained="bert-base-uncased",
                d_model=768,
                config=str(_CONFIGS_DIR / "config_bert.json"),
            ),
            temporal_modeling=dict(
                enabled=True,
                temporal_model_block="timesformer",
                temporal_model_position="last",
                temporal_model_config=dict(input_dim=768),
                use_temporal_position_embedding=True,
            ),
        ),
    ),
}


def get_config(model_name: str, overrides: dict = None) -> SimpleNamespace:
    if model_name not in _MODEL_DEFAULTS:
        raise ValueError(f"No config for '{model_name}'. Available: {list(_MODEL_DEFAULTS)}")
    cfg = copy.deepcopy(_MODEL_DEFAULTS[model_name])
    if overrides:
        _deep_update(cfg, overrides)
    return _dict_to_ns(cfg)