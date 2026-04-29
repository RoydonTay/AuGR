"""
Mixture-of-Experts (MoE) FFN Layer

Sparse MoE layer with top-k gating and load-balancing auxiliary loss.
Designed as a drop-in replacement for a standard FFN block.

Reference: Switch Transformer (Fedus et al., 2022), ST-MoE (Zoph et al., 2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """Sparse Mixture-of-Experts Feed-Forward Network.

    Each token is routed to top-k experts via a learned gating network.
    Includes a load-balancing auxiliary loss to prevent expert collapse.

    Args:
        embed_dim: Input/output dimension.
        ffn_dim: Hidden dimension of each expert FFN.
        num_experts: Number of expert FFN modules.
        top_k: Number of experts activated per token.
        dropout: Dropout rate inside each expert.
        load_balance_weight: Coefficient for the load-balancing aux loss.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int = 0,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        if ffn_dim <= 0:
            ffn_dim = 4 * embed_dim

        # Gating network: token repr → expert scores
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

        # Expert FFN modules (each is a small 2-layer MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        # LayerNorm + residual
        self.norm = nn.LayerNorm(embed_dim)

        # Routing diagnostics (updated every forward)
        self._routing_stats = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (*, embed_dim) — input tokens (any leading batch dims).

        Returns:
            output: (*, embed_dim) — MoE output with residual connection.
            aux_loss: scalar — load-balancing loss.
        """
        orig_shape = x.shape
        # Flatten to (T, D) where T = product of all leading dims
        x_flat = x.reshape(-1, self.embed_dim)  # (T, D)
        T = x_flat.size(0)

        # --- Gating ---
        gate_logits = self.gate(x_flat)              # (T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (T, E)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # (T, K)
        # Normalize top-k weights to sum to 1
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)  # (T, K)

        # --- Expert computation ---
        # For small num_experts, loop is efficient and avoids complex scatter/gather
        expert_outputs = torch.zeros_like(x_flat)  # (T, D)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]     # (T,)
            weight = topk_weights[:, k]     # (T,)
            for e in range(self.num_experts):
                mask = (expert_idx == e)    # (T,) bool
                if mask.any():
                    expert_input = x_flat[mask]                          # (n, D)
                    expert_out = self.experts[e](expert_input)           # (n, D)
                    expert_outputs[mask] += weight[mask].unsqueeze(-1) * expert_out

        # Residual + LayerNorm
        output = self.norm(x_flat + expert_outputs)

        # --- Load-balancing auxiliary loss ---
        # Encourages uniform expert utilization (Switch Transformer style)
        aux_loss = self._load_balance_loss(gate_probs, topk_idx, T)

        # --- Routing diagnostics ---
        self._update_routing_stats(gate_probs, topk_idx, T)

        return output.reshape(orig_shape), aux_loss

    def _load_balance_loss(self, gate_probs, topk_idx, T):
        """Compute load-balancing loss (Switch Transformer Eq. 4).

        L_balance = E * sum_e(f_e * P_e)
        where:
            f_e = fraction of tokens routed to expert e
            P_e = mean gate probability for expert e
        """
        # f_e: fraction of tokens dispatched to each expert
        # Use one-hot of all top-k selections
        one_hot = F.one_hot(topk_idx, self.num_experts).float()  # (T, K, E)
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)        # (E,)
        f = tokens_per_expert / (T * self.top_k)                 # (E,)

        # P_e: mean gate probability over all tokens for each expert
        P = gate_probs.mean(dim=0)  # (E,)

        # L = num_experts * sum(f_e * P_e)
        loss = self.load_balance_weight * self.num_experts * (f * P).sum()
        return loss

    def _update_routing_stats(self, gate_probs, topk_idx, T):
        """Store routing diagnostics for external logging."""
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, self.num_experts).float()
            tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # (E,)
            f = tokens_per_expert / (T * self.top_k)
            P = gate_probs.mean(dim=0)
            # Entropy of the routing distribution (higher = more uniform)
            entropy = -(f * (f + 1e-9).log()).sum()
            max_entropy = torch.tensor(self.num_experts, dtype=f.dtype).log()
            self._routing_stats = {
                'expert_frac': f.detach().cpu(),       # (E,) fraction of tokens per expert
                'expert_gate_prob': P.detach().cpu(),  # (E,) mean gate probability
                'routing_entropy': entropy.item(),     # scalar
                'max_entropy': max_entropy.item(),     # scalar (ln E)
                'num_tokens': T,
            }


class MoEFFNDropIn(nn.Module):
    """Drop-in MoE replacement for a standard FFN (nn.Sequential).

    Unlike MoEFFN, this does NOT include residual connection or LayerNorm,
    so it can directly replace HSTULayer.ffn while the layer handles
    residual and pre-norm externally: ``x = x + ffn(ffn_norm(x))``.

    After each forward call, the load-balancing auxiliary loss is stored
    in ``self._aux_loss`` for retrieval by the parent model.

    Args:
        embed_dim: Input/output dimension.
        ffn_dim: Hidden dimension of each expert FFN (0 = 4*embed_dim).
        num_experts: Number of expert FFN modules.
        top_k: Number of experts activated per token.
        dropout: Dropout rate inside each expert.
        load_balance_weight: Coefficient for the load-balancing aux loss.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int = 0,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        if ffn_dim <= 0:
            ffn_dim = 4 * embed_dim

        # Gating network
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

        # Expert FFNs (same structure as HSTULayer.ffn)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        self._aux_loss = torch.tensor(0.0)

        # Routing diagnostics (updated every forward)
        self._routing_stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns output tensor only (no residual).

        Aux loss is stored in ``self._aux_loss``.

        Args:
            x: (*, embed_dim) — pre-normed input from HSTULayer.

        Returns:
            output: (*, embed_dim) — same shape as input.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.embed_dim)  # (T, D)
        T = x_flat.size(0)

        # --- Gating ---
        gate_logits = self.gate(x_flat)              # (T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (T, E)

        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # (T, K)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # --- Expert computation ---
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]
            weight = topk_weights[:, k]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += weight[mask].unsqueeze(-1) * expert_out

        # --- Store aux loss as attribute ---
        one_hot = F.one_hot(topk_idx, self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)
        f = tokens_per_expert / (T * self.top_k)
        P = gate_probs.mean(dim=0)
        self._aux_loss = self.load_balance_weight * self.num_experts * (f * P).sum()

        # --- Routing diagnostics ---
        with torch.no_grad():
            entropy = -(f * (f + 1e-9).log()).sum()
            max_entropy = torch.tensor(self.num_experts, dtype=f.dtype).log()
            self._routing_stats = {
                'expert_frac': f.detach().cpu(),
                'expert_gate_prob': P.detach().cpu(),
                'routing_entropy': entropy.item(),
                'max_entropy': max_entropy.item(),
                'num_tokens': T,
            }

        return output.reshape(orig_shape)
