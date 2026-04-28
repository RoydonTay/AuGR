from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from .moe_layer import MoEFFN
from ..models.hstu import HSTULayer
from intentrcmd.modules.custom_loss import calc_aux_loss_weight


class AvazuUniGCRConfig(PretrainedConfig):
    """Config for Avazu unified CTR + generative model."""

    model_type = "avazu-unigcr"

    def __init__(
        self,
        feature_vocab_sizes: Optional[Dict[str, int]] = None,
        user_feature_names: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        c14_feature_name: str = "C14",
        emb_size: int = 128,
        d_model: Optional[int] = None,
        dropout: float = 0.2,
        ctr_hidden_units: List[int] = None,
        hstu_num_heads: int = 2,
        hstu_num_blocks: int = 4,
        hstu_num_position_buckets: int = 32,
        hstu_num_time_buckets: int = 64,
        hstu_max_position_distance: int = 128,
        lambda_ctr: float = 1.0,
        lambda_gen: float = 0.1,
        ctr_shallow_shortcut: bool = False,
        gen_loss_decay: bool = True,
        use_post_hstu_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_load_balance_weight: float = 0.01,
        moe_ffn_dim: int = 0,
        **kwargs,
    ):
        # Must allow zero-arg construction for HF internals during save_pretrained.
        self.feature_vocab_sizes = feature_vocab_sizes or {}
        self.user_feature_names = user_feature_names or []
        self.feature_groups = feature_groups or {}
        self.c14_feature_name = c14_feature_name
        self.emb_size = emb_size
        self.d_model = d_model if d_model is not None else emb_size
        self.dropout = dropout
        self.ctr_hidden_units = ctr_hidden_units or [256, 128]
        self.hstu_num_heads = hstu_num_heads
        self.hstu_num_blocks = hstu_num_blocks
        self.hstu_num_position_buckets = hstu_num_position_buckets
        self.hstu_num_time_buckets = hstu_num_time_buckets
        self.hstu_max_position_distance = hstu_max_position_distance
        self.lambda_ctr = lambda_ctr
        self.lambda_gen = lambda_gen
        self.ctr_shallow_shortcut = ctr_shallow_shortcut
        self.gen_loss_decay = gen_loss_decay
        self.use_post_hstu_moe = use_post_hstu_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_load_balance_weight = moe_load_balance_weight
        self.moe_ffn_dim = moe_ffn_dim
        super().__init__(**kwargs)


class AvazuUniGCRModel(PreTrainedModel):
    """Avazu adaptation of UniGCR-style mixed-mask sequence modeling.

    - User tokens: all categorical features except C14.
    - Intent-like token: C14.
    - CTR head: BCE on click labels.
    - Generative head: predict C14 id from h_user.
    """

    config_class = AvazuUniGCRConfig

    def __init__(self, config: AvazuUniGCRConfig):
        super().__init__(config)
        self.emb_size = config.emb_size
        self.d_model = config.d_model
        self.lambda_ctr = config.lambda_ctr
        self.lambda_gen = config.lambda_gen
        self.ctr_shallow_shortcut = config.ctr_shallow_shortcut
        self.gen_loss_decay = config.gen_loss_decay
        self.use_post_hstu_moe = config.use_post_hstu_moe

        # Step tracking used by StepUpdateCallback for auxiliary loss decay.
        self.max_steps = 0
        self.global_step = 0

        # Per-feature sparse embeddings.
        embeddings = {}
        for name, vocab_size in config.feature_vocab_sizes.items():
            embeddings[name] = nn.Embedding(vocab_size, config.emb_size)
        self.feature_embeddings = nn.ModuleDict(embeddings)

        self.user_feature_groups = self._resolve_user_feature_groups(
            user_feature_names=config.user_feature_names,
            feature_groups=config.feature_groups,
        )
        self.group_feature_dropout = nn.ModuleDict({
            group_name: nn.ModuleDict({
                feature_name: nn.Dropout(config.dropout)
                for feature_name in feature_list
            })
            for group_name, feature_list in self.user_feature_groups.items()
        })
        self.group_output_encoders = nn.ModuleDict()
        self.group_output_projections = nn.ModuleDict()
        for group_name, feature_list in self.user_feature_groups.items():
            group_dim = len(feature_list) * config.emb_size
            self.group_output_encoders[group_name] = nn.Sequential(
                nn.Linear(group_dim, group_dim),
                nn.BatchNorm1d(group_dim),
                nn.PReLU(),
                nn.Dropout(config.dropout),
            )
            self.group_output_projections[group_name] = nn.Sequential(
                nn.Linear(group_dim, config.d_model),
                nn.PReLU(),
            )

        self.input_dropout = nn.Dropout(config.dropout)

        self.hstu_layers = nn.ModuleList([
            HSTULayer(
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
        self.hstu_final_norm = nn.LayerNorm(config.d_model)

        self.c14_projection = nn.Sequential(
            nn.Linear(config.emb_size, config.d_model),
            nn.BatchNorm1d(config.d_model),
            nn.PReLU(),
            nn.Dropout(config.dropout),
        )

        # Post HSTU MoE on C14 token
        if self.use_post_hstu_moe:
            self.c14_moe = MoEFFN(
                    embed_dim=config.d_model,
                    ffn_dim=config.moe_ffn_dim,
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k,
                    dropout=config.dropout,
                    load_balance_weight=config.moe_load_balance_weight,
                )

        # CTR HEAD
        ctr_input_dim = config.d_model  # h_c14_hstu
        if self.ctr_shallow_shortcut:
            # concat h_user_hstu with all user feature embs
            shallow_dim = config.d_model * len(self.user_feature_groups)
            shallow_hidden = min(2048, shallow_dim // 2)
            self.shallow_interaction = nn.Sequential(
                nn.Linear(shallow_dim, shallow_hidden),
                nn.BatchNorm1d(shallow_hidden),
                nn.PReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(shallow_hidden, config.d_model),
            )
            self.shallow_output_act = nn.PReLU()
            ctr_input_dim += config.d_model

        # CTR: use both deep C14 token and user summary.
        self.ctr_tower = nn.Sequential(
            self._build_mlp(ctr_input_dim, config.ctr_hidden_units, config.dropout),
            nn.Linear(config.ctr_hidden_units[-1], 1),
        )

        # Generative head predicts C14 id.
        c14_vocab_size = config.feature_vocab_sizes[config.c14_feature_name]
        self.gen_head = nn.Linear(config.d_model, c14_vocab_size)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_units: List[int], dropout: float):
        layers = []
        in_dim = input_dim
        for h in hidden_units:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.PReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        return nn.Sequential(*layers)

    @staticmethod
    def _resolve_user_feature_groups(user_feature_names: List[str], feature_groups: Dict[str, List[str]]):
        """Resolve grouping config into ordered group -> features mapping.

        If no grouping config is provided, each sparse feature becomes one token.
        """
        if not feature_groups:
            return {name: [name] for name in user_feature_names}

        user_feature_set = set(user_feature_names)
        resolved_groups = {}
        used_features = set()

        for group_name, feature_list in feature_groups.items():
            if not isinstance(feature_list, list) or len(feature_list) == 0:
                continue

            normalized = []
            for f in feature_list:
                # Backward-compatible alias: weekend -> weekday for Avazu time feature.
                if f not in user_feature_set:
                    raise ValueError(f"Unknown user feature in group {group_name}: {f}")
                normalized.append(f)

            # Keep order while removing duplicates inside one group (in case of error in config)
            seen = set()
            deduped = []
            for f in normalized:
                if f in seen:
                    continue
                deduped.append(f)
                seen.add(f)

            resolved_groups[group_name] = deduped
            used_features.update(deduped)

        # Ensure every user feature is covered at least once.
        missing = [f for f in user_feature_names if f not in used_features]
        for f in missing:
            resolved_groups[f] = [f]

        return resolved_groups
    
    def _shallow_encode(self, user_tokens: torch.Tensor):
        """Per-group MLP + cross-group interaction"""
        concat_user_tokens = user_tokens.view(user_tokens.size(0), -1)  # (B, N_user * D)
        h_shallow = self.shallow_interaction(concat_user_tokens)
        h_shallow = self.shallow_output_act(h_shallow)
        return h_shallow

    def _build_user_tokens(self, kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Build one token per feature-group.
        # Group encoder follows UserSoloEncoder pattern: emb -> dropout -> concat -> output mlp.
        group_tokens = []
        for group_name, feature_list in self.user_feature_groups.items():
            feat_embeddings = []
            for feature_name in feature_list:
                v = kwargs[feature_name].long().squeeze(-1)
                num_embeddings = self.config.feature_vocab_sizes[feature_name]
                v = torch.where(v < 0, 0, v)
                v = torch.where(v >= num_embeddings, 0, v)

                emb = self.feature_embeddings[feature_name](v)
                emb = self.group_feature_dropout[group_name][feature_name](emb)
                feat_embeddings.append(emb)

            group_concat = torch.cat(feat_embeddings, dim=-1)
            group_encoded = self.group_output_encoders[group_name](group_concat)
            token = self.group_output_projections[group_name](group_encoded)
            group_tokens.append(token.unsqueeze(1))

        return torch.cat(group_tokens, dim=1)

    def _build_c14_token(self, kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        c14_name = self.config.c14_feature_name
        v = kwargs[c14_name].long().squeeze(-1)
        emb = self.feature_embeddings[c14_name](v)
        token = self.c14_projection(emb)
        return token.unsqueeze(1)

    @staticmethod
    def _build_mixed_mask(n_user: int, n_intent: int, device):
        """MTGR-style mixed mask: user-user visible, intent->user visible only."""
        total_n = n_user + n_intent
        mask = torch.ones(total_n, total_n, device=device, dtype=torch.bool)
        mask[:n_user, :n_user] = False
        mask[n_user:, :n_user] = False
        return mask

    def _forward_hstu(self, user_tokens: torch.Tensor, c14_token: torch.Tensor):
        x = torch.cat([user_tokens, c14_token], dim=1)
        x = self.input_dropout(x)

        bsz, n_total, _ = x.shape
        n_user = user_tokens.size(1)
        mixed_mask = self._build_mixed_mask(n_user=n_user, n_intent=1, device=x.device)
        padding_mask = torch.zeros(bsz, n_total, device=x.device, dtype=torch.bool)

        for layer in self.hstu_layers:
            x = layer(x, mixed_mask, padding_mask)
        x = self.hstu_final_norm(x)

        h_user = x[:, n_user - 1, :]
        h_c14 = x[:, n_user, :]
        return h_user, h_c14

    def forward(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            kwargs = args[0]

        labels_click = kwargs.get("labels_click", None)
        c14_name = self.config.c14_feature_name
        labels_c14 = kwargs.get("labels_c14", kwargs.get(c14_name, None))

        user_tokens = self._build_user_tokens(kwargs) # (B, N_user, D)
        c14_token = self._build_c14_token(kwargs)
        h_user, h_c14 = self._forward_hstu(user_tokens, c14_token)

        moe_load_balance_loss = torch.tensor(0.0, device=h_c14.device)
        if self.use_post_hstu_moe:
            h_c14, moe_load_balance_loss = self.c14_moe(h_c14)


        if self.ctr_shallow_shortcut:
            h_shallow = self._shallow_encode(user_tokens)
            ctr_input = torch.cat([h_c14, h_shallow], dim=-1)
        else:
            #ctr_input = torch.cat([h_c14, h_user], dim=-1)
            ctr_input = h_c14
        logits_click = self.ctr_tower(ctr_input).squeeze(-1)
        gen_logits = self.gen_head(h_user)

        if labels_click is None:
            return torch.sigmoid(logits_click)

        click_targets = labels_click.float().view(-1)
        ctr_loss = F.binary_cross_entropy_with_logits(logits_click, click_targets)

        gen_loss = torch.tensor(0.0, device=logits_click.device)
        if labels_c14 is not None:
            c14_targets = labels_c14.long().view(-1)
            gen_loss = F.cross_entropy(gen_logits, c14_targets, ignore_index=0)

        gen_weight = self.lambda_gen
        if self.gen_loss_decay:
            gen_weight *= calc_aux_loss_weight(self.global_step, self.max_steps)

        total_loss = self.lambda_ctr * ctr_loss + gen_weight * gen_loss + moe_load_balance_loss
        return {
            "loss": total_loss,
            "logits": logits_click,
            "ctr_loss": ctr_loss,
            "gen_loss": gen_loss,
            "gen_weight": torch.tensor(gen_weight, device=logits_click.device),
            "logits_click": logits_click,
            "gen_logits": gen_logits,
        }
