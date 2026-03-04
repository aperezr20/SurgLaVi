import torch
from typing import Union

from .bert_tokenizer import BertTokenizer


def load_tokenizer(config) -> BertTokenizer:
    return BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)


def tokenize(
    texts: Union[str, list],
    tokenizer: BertTokenizer,
    max_length: int = 77,
    device: Union[str, torch.device] = "cpu",
    truncate: bool = True,
):
    if isinstance(texts, str):
        texts = [texts]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=truncate,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)