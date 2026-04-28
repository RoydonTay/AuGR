"""HSTU layer variant for full or custom self-attention masking."""

from typing import Optional

import torch

from .hstu import HSTULayer


class HSTUFullSelfAttentionLayer(HSTULayer):
    """HSTU layer with explicit full user-token self-attention behavior.

    Behavioral change vs ``HSTULayer``:
    - The user-user block is always unmasked (non-causal) when
      ``n_user_tokens`` is provided.
    - Optional ``attention_mask`` still controls all non user-user pairs
      (for example, preserving C14 visibility constraints).
    """

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        padding_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        n_user_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        seq_len = x.size(1)
        if attention_mask is None:
            # Default to full attention; all masking comes from padding.
            effective_mask = torch.zeros(seq_len, seq_len, device=x.device, dtype=torch.bool)
        else:
            effective_mask = attention_mask.clone()

        if n_user_tokens is not None:
            if n_user_tokens < 0 or n_user_tokens > seq_len:
                raise ValueError(f"n_user_tokens must be in [0, {seq_len}], got {n_user_tokens}")
            # Force full (non-causal) attention among user tokens.
            effective_mask[:n_user_tokens, :n_user_tokens] = False

        return super().forward(
            x=x,
            causal_mask=effective_mask,
            padding_mask=padding_mask,
            timestamps=timestamps,
        )
