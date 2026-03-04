import json
import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel

from .models.timesformer.timesformer import TimeSformer

logger = logging.getLogger(__name__)


class SurgCLIP(nn.Module):

    def __init__(self, config, tokenizer, is_pretrain: bool = True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        self.vision_width = config.model.vision_encoder.d_model
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        self.vision_encoder, self.vision_layernorm = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config.model.temp)

    def encode_vision(self, image):
        image = image.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        vision_embeds, pooled_vision_embeds = self.vision_encoder(image)
        vision_embeds = self.vision_layernorm(vision_embeds)
        return vision_embeds, pooled_vision_embeds

    def encode_text(self, text):
        encoder = self.get_text_encoder()
        output = encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_embeds = output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    def forward(self, video: torch.Tensor, text):
        _, pooled_vision = self.encode_vision(video)
        _, pooled_text = self.encode_text(text)

        sim_v2t, sim_t2v = self.get_sim(
            self.vision_proj(pooled_vision),
            self.text_proj(pooled_text),
            temp=self.temp,
        )
        return sim_v2t, sim_t2v

    def get_text_encoder(self):
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def build_vision_encoder(self):
        cfg = json.load(open(self.config.model.vision_encoder.config))
        if self.config.model.temporal_modeling.enabled:
            cfg.update({"num_frames": self.config.num_frames, "attention_type": "divided_space_time"})
        else:
            cfg.update({"num_frames": 1, "attention_type": "space_only"})
        cfg.update({"num_classes": 0, "gradient_checkpointing": self.config.gradient_checkpointing})

        vision_encoder = TimeSformer(**cfg)
        vision_layernorm = (
            nn.LayerNorm(self.vision_width, eps=1e-12)
            if self.config.model.vit_add_ln else nn.Identity()
        )
        return vision_encoder, vision_layernorm

    def build_text_encoder(self):
        bert_config = BertConfig.from_json_file(self.config.model.text_encoder.config)
        bert_config.encoder_width = self.config.model.vision_encoder.d_model
        bert_config.gradient_checkpointing = self.config.gradient_checkpointing

        text_encoder, _ = BertModel.from_pretrained(
            self.config.model.text_encoder.pretrained,
            config=bert_config,
            add_pooling_layer=False,
            output_loading_info=True,
        )
        return text_encoder

    @staticmethod
    def get_sim(
        vision_proj: torch.Tensor,
        text_proj: torch.Tensor,
        temp=1.0,
    ):
        """calculate pair-wise video-text similarity.

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            temp (torch.Tensor): The temperature. Shape: [].

        Returns: The similarity between video and text. Shape: [B,B].

        """
        vision_proj = F.normalize(vision_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        if len(vision_proj.shape) == 3: #(b, t, c)
            sim_v2t = torch.einsum("mld,nd->mln", vision_proj, text_proj).mean(1) / temp  # (B,B)

        elif len(vision_proj.shape) == 2: #(b, c)
            sim_v2t = torch.einsum("md,nd->mn", vision_proj, text_proj) / temp  # (B,B)

        sim_t2v = sim_v2t.T
        return sim_v2t, sim_t2v