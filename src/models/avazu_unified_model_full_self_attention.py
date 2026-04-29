from typing import Dict

import torch
from torch import nn

from .hstu_full_self_attention import HSTUFullSelfAttentionLayer
from .avazu_unified_model import AvazuUniGCRConfig, AvazuUniGCRModel


class AvazuUniGCRFullSelfAttentionModel(AvazuUniGCRModel):
    """Avazu UniGCR variant with full user-token self-attention and mean pooling.

    Changes from AvazuUniGCRModel:
    1. Uses a dedicated full-attention HSTU layer wrapper.
    2. Uses mean pooling over user token hidden states for generative head input.
    3. Keeps C14 masking behavior unchanged via the same mixed mask logic.
    """

    config_class = AvazuUniGCRConfig

    def __init__(self, config: AvazuUniGCRConfig):
        super().__init__(config)

        self.hstu_layers = nn.ModuleList([
            HSTUFullSelfAttentionLayer(
                embed_dim=config.d_model,
                num_heads=config.hstu_num_heads,
                dropout=config.dropout,
                num_position_buckets=config.hstu_num_position_buckets,
                num_time_buckets=config.hstu_num_time_buckets,
                max_position_distance=config.hstu_max_position_distance,
                use_temporal_bias=False,
            )
            for _ in range(config.hstu_num_blocks)
        ])

    def _forward_hstu(self, user_tokens: torch.Tensor, c14_token: torch.Tensor):
        x = torch.cat([user_tokens, c14_token], dim=1)
        x = self.input_dropout(x)

        bsz, n_total, _ = x.shape
        n_user = user_tokens.size(1)

        # Keep C14 treatment unchanged:
        # - user tokens can only attend to user tokens (not C14)
        # - C14 token can attend to user tokens
        mixed_mask = self._build_mixed_mask(n_user=n_user, n_intent=1, device=x.device)
        padding_mask = torch.zeros(bsz, n_total, device=x.device, dtype=torch.bool)

        for layer in self.hstu_layers:
            x = layer(
                x,
                attention_mask=mixed_mask,
                padding_mask=padding_mask,
                n_user_tokens=n_user,
            )

        x = self.hstu_final_norm(x)

        user_hidden_states = x[:, :n_user, :]
        h_user = user_hidden_states.mean(dim=1)
        h_c14 = x[:, n_user, :]
        return h_user, h_c14
