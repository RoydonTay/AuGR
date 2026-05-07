from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from intentrcmd.modules.pooling_layer import MultiheadAttentionPooling, SelfAttentionPooling
from intentrcmd.modules.custom_loss import calc_aux_loss_weight

from .moe_layer import MoEFFN


POOLING_CLS = {
    "self_attention": SelfAttentionPooling,
    "multihead_attention": MultiheadAttentionPooling,
}


class TaobaoUserSeqEncoder(nn.Module):
    """Sequence feature encoder aligned with Taobao unified model behavior."""

    def __init__(
        self,
        sequence_feature_names: List[str],
        feature_embeddings: nn.ModuleDict,
        feature_vocab_sizes: Dict[str, int],
        emb_size: int,
        d_model: int,
        dropout: float = 0.2,
        pooling_type: str = "self_attention",
    ):
        super().__init__()
        if pooling_type not in POOLING_CLS:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        self.sequence_feature_names = sequence_feature_names
        self.feature_embeddings = feature_embeddings
        self.feature_vocab_sizes = feature_vocab_sizes

        self.dropout_layers = nn.ModuleDict({
            name: nn.Dropout(dropout) for name in sequence_feature_names
        })
        self.pooling_layers = nn.ModuleDict({
            name: POOLING_CLS[pooling_type](hidden_size=emb_size, dropout=dropout)
            for name in sequence_feature_names
        })
        self.output_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(emb_size, d_model),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
            for name in sequence_feature_names
        })

    def forward(self, kwargs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        seq_tokens = []
        for name in self.sequence_feature_names:
            if name not in kwargs:
                raise ValueError(f"Missing sequence feature in model inputs: {name}")

            values = kwargs[name].long()
            if values.dim() == 1:
                values = values.unsqueeze(1)
            num_embeddings = self.feature_vocab_sizes[name]
            values = torch.where(values < 0, 0, values)
            values = torch.where(values >= num_embeddings, 0, values)

            emb = self.feature_embeddings[name](values)
            emb = self.dropout_layers[name](emb)
            mask = (values != 0).float()
            pooled = self.pooling_layers[name](emb, mask)
            seq_token = self.output_projections[name](pooled).unsqueeze(1)
            seq_tokens.append(seq_token)

        if not seq_tokens:
            return None
        return torch.cat(seq_tokens, dim=1)


class TaobaoSASRecGenConfig(PretrainedConfig):
    """TaoBao SASRec-based config with CTR + generative head."""

    model_type = "taobao-sasrec-gen"

    def __init__(
        self,
        feature_vocab_sizes: Optional[Dict[str, int]] = None,
        user_feature_names: Optional[List[str]] = None,
        item_feature_names: Optional[List[str]] = None,
        sequence_feature_names: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        target_feature_name: str = "cate_id",
        emb_size: int = 128,
        d_model: Optional[int] = None,
        dropout: float = 0.2,
        sequence_pooling_type: str = "self_attention",
        ctr_hidden_units: List[int] = None,
        sasrec_num_heads: int = 2,
        sasrec_num_blocks: int = 2,
        sasrec_ffn_dim: int = 256,
        lambda_ctr: float = 1.0,
        lambda_gen: float = 0.1,
        ctr_shallow_shortcut: bool = False,
        gen_loss_decay: bool = True,
        use_post_sasrec_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_load_balance_weight: float = 0.01,
        moe_ffn_dim: int = 0,
        **kwargs,
    ):
        self.feature_vocab_sizes = feature_vocab_sizes or {}
        self.user_feature_names = user_feature_names or []
        self.item_feature_names = item_feature_names or [target_feature_name]
        self.sequence_feature_names = sequence_feature_names or []
        self.feature_groups = feature_groups or {}
        self.target_feature_name = target_feature_name
        self.emb_size = emb_size
        self.d_model = d_model if d_model is not None else emb_size
        self.dropout = dropout
        self.sequence_pooling_type = sequence_pooling_type
        self.ctr_hidden_units = ctr_hidden_units or [256, 128]
        self.sasrec_num_heads = sasrec_num_heads
        self.sasrec_num_blocks = sasrec_num_blocks
        self.sasrec_ffn_dim = sasrec_ffn_dim
        self.lambda_ctr = lambda_ctr
        self.lambda_gen = lambda_gen
        self.ctr_shallow_shortcut = ctr_shallow_shortcut
        self.gen_loss_decay = gen_loss_decay
        self.use_post_sasrec_moe = use_post_sasrec_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_load_balance_weight = moe_load_balance_weight
        self.moe_ffn_dim = moe_ffn_dim
        super().__init__(**kwargs)


class TaobaoSASRecGenModel(PreTrainedModel):
    """TaoBao SASRec backbone with CTR head + generative target head."""

    config_class = TaobaoSASRecGenConfig

    def __init__(self, config: TaobaoSASRecGenConfig):
        super().__init__(config)
        self.emb_size = config.emb_size
        self.d_model = config.d_model
        self.lambda_ctr = config.lambda_ctr
        self.lambda_gen = config.lambda_gen
        self.ctr_shallow_shortcut = config.ctr_shallow_shortcut
        self.gen_loss_decay = config.gen_loss_decay
        self.use_post_sasrec_moe = config.use_post_sasrec_moe

        # Step tracking used by StepUpdateCallback for auxiliary-loss decay.
        self.max_steps = 0
        self.global_step = 0

        embeddings = {}
        for name, vocab_size in config.feature_vocab_sizes.items():
            embeddings[name] = nn.Embedding(vocab_size, config.emb_size)
        self.feature_embeddings = nn.ModuleDict(embeddings)

        self.user_feature_groups = self._resolve_feature_groups(
            feature_names=config.user_feature_names,
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

        self.sequence_feature_names = config.sequence_feature_names
        self.item_feature_names = config.item_feature_names
        if not self.item_feature_names:
            raise ValueError("item_feature_names must contain at least one feature")
        for feature_name in self.item_feature_names:
            if feature_name not in config.feature_vocab_sizes:
                raise ValueError(f"item feature is missing from feature_vocab_sizes: {feature_name}")

        self.user_seq_encoder = None
        if self.sequence_feature_names:
            self.user_seq_encoder = TaobaoUserSeqEncoder(
                sequence_feature_names=self.sequence_feature_names,
                feature_embeddings=self.feature_embeddings,
                feature_vocab_sizes=config.feature_vocab_sizes,
                emb_size=config.emb_size,
                d_model=config.d_model,
                dropout=config.dropout,
                pooling_type=config.sequence_pooling_type,
            )
            self.user_seq_encoder._tied_weights_keys = {
                r"feature_embeddings\..*\.weight": "feature_embeddings"
            }

        self.total_user_tokens = len(self.user_feature_groups) + len(self.sequence_feature_names)

        self.input_dropout = nn.Dropout(config.dropout)

        self.sasrec_blocks = nn.ModuleList([
            SASRecBlock(
                embed_dim=config.d_model,
                num_heads=config.sasrec_num_heads,
                ffn_dim=config.sasrec_ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.sasrec_num_blocks)
        ])
        self.sasrec_final_norm = nn.LayerNorm(config.d_model)

        self.item_feature_dropouts = nn.ModuleDict({
            name: nn.Dropout(config.dropout) for name in self.item_feature_names
        })
        self.item_feature_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(config.emb_size, config.d_model),
                nn.BatchNorm1d(config.d_model),
                nn.PReLU(),
                nn.Dropout(config.dropout),
            )
            for name in self.item_feature_names
        })

        if self.use_post_sasrec_moe:
            self.target_moe = MoEFFN(
                embed_dim=config.d_model,
                ffn_dim=config.moe_ffn_dim,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                dropout=config.dropout,
                load_balance_weight=config.moe_load_balance_weight,
            )

        ctr_input_dim = config.d_model
        if self.ctr_shallow_shortcut:
            shallow_dim = config.d_model * self.total_user_tokens
            shallow_hidden = min(2048, max(shallow_dim // 2, config.d_model))
            self.shallow_interaction = nn.Sequential(
                nn.Linear(shallow_dim, shallow_hidden),
                nn.BatchNorm1d(shallow_hidden),
                nn.PReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(shallow_hidden, config.d_model),
            )
            self.shallow_output_act = nn.PReLU()
            ctr_input_dim += config.d_model

        self.ctr_tower = nn.Sequential(
            self._build_mlp(ctr_input_dim, config.ctr_hidden_units, config.dropout),
            nn.Linear(config.ctr_hidden_units[-1], 1),
        )

        target_vocab_size = config.feature_vocab_sizes[config.target_feature_name]
        self.gen_head = nn.Linear(config.d_model, target_vocab_size)

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
    def _resolve_feature_groups(feature_names: List[str], feature_groups: Dict[str, List[str]]):
        if not feature_names:
            return {}
        if not feature_groups:
            return {name: [name] for name in feature_names}

        feature_set = set(feature_names)
        resolved_groups = {}
        used_features = set()

        for group_name, feature_list in feature_groups.items():
            if not isinstance(feature_list, list) or len(feature_list) == 0:
                continue

            normalized = []
            for feature_name in feature_list:
                if feature_name not in feature_set:
                    continue
                normalized.append(feature_name)
            if not normalized:
                continue

            seen = set()
            deduped = []
            for feature_name in normalized:
                if feature_name in seen:
                    continue
                deduped.append(feature_name)
                seen.add(feature_name)

            resolved_groups[group_name] = deduped
            used_features.update(deduped)

        missing = [name for name in feature_names if name not in used_features]
        for name in missing:
            resolved_groups[name] = [name]

        return resolved_groups

    def _shallow_encode(self, user_tokens: torch.Tensor):
        concat_user_tokens = user_tokens.view(user_tokens.size(0), -1)
        h_shallow = self.shallow_interaction(concat_user_tokens)
        h_shallow = self.shallow_output_act(h_shallow)
        return h_shallow

    def _build_user_tokens(self, kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        group_tokens = []
        for group_name, feature_list in self.user_feature_groups.items():
            feat_embeddings = []
            for feature_name in feature_list:
                values = kwargs[feature_name].long().squeeze(-1)
                num_embeddings = self.config.feature_vocab_sizes[feature_name]
                values = torch.where(values < 0, 0, values)
                values = torch.where(values >= num_embeddings, 0, values)

                emb = self.feature_embeddings[feature_name](values)
                emb = self.group_feature_dropout[group_name][feature_name](emb)
                feat_embeddings.append(emb)

            group_concat = torch.cat(feat_embeddings, dim=-1)
            group_encoded = self.group_output_encoders[group_name](group_concat)
            token = self.group_output_projections[group_name](group_encoded)
            group_tokens.append(token.unsqueeze(1))

        if self.user_seq_encoder is not None:
            seq_tokens = self.user_seq_encoder(kwargs)
            if seq_tokens is not None:
                group_tokens.append(seq_tokens)

        if not group_tokens:
            raise ValueError("No user tokens were constructed; check feature configuration")
        return torch.cat(group_tokens, dim=1)

    def _build_item_tokens(self, kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_tokens = []
        for feature_name in self.item_feature_names:
            if feature_name not in kwargs:
                raise ValueError(f"Missing item feature in model inputs: {feature_name}")

            values = kwargs[feature_name].long().squeeze(-1)
            num_embeddings = self.config.feature_vocab_sizes[feature_name]
            values = torch.where(values < 0, 0, values)
            values = torch.where(values >= num_embeddings, 0, values)

            emb = self.feature_embeddings[feature_name](values)
            emb = self.item_feature_dropouts[feature_name](emb)
            token = self.item_feature_projections[feature_name](emb)
            item_tokens.append(token.unsqueeze(1))

        return torch.cat(item_tokens, dim=1)

    @staticmethod
    def _build_mixed_mask(n_user: int, n_item: int, device):
        total_n = n_user + n_item
        mask = torch.ones(total_n, total_n, device=device, dtype=torch.bool)
        mask[:n_user, :n_user] = False
        mask[n_user:, :n_user] = False
        return mask

    def _forward_sasrec(self, user_tokens: torch.Tensor, item_tokens: torch.Tensor):
        x = torch.cat([user_tokens, item_tokens], dim=1)
        x = self.input_dropout(x)

        bsz, n_total, _ = x.shape
        n_user = user_tokens.size(1)
        n_item = item_tokens.size(1)
        mixed_mask = self._build_mixed_mask(n_user=n_user, n_item=n_item, device=x.device)
        padding_mask = torch.zeros(bsz, n_total, device=x.device, dtype=torch.bool)

        for block in self.sasrec_blocks:
            x = block(x, padding_mask, mixed_mask)
        x = self.sasrec_final_norm(x)

        h_user = x[:, n_user - 1, :]
        item_hidden_states = x[:, n_user:, :]
        h_item = item_hidden_states.mean(dim=1)

        moe_load_balance_loss = torch.tensor(0.0, device=x.device)
        if self.use_post_sasrec_moe:
            h_item, moe_load_balance_loss = self.target_moe(h_item)

        return h_user, h_item, moe_load_balance_loss

    def forward(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            kwargs = args[0]

        labels_click = kwargs.get("labels_click", None)
        target_name = self.config.target_feature_name
        labels_target = kwargs.get("labels_target", kwargs.get(target_name, None))

        user_tokens = self._build_user_tokens(kwargs)
        item_tokens = self._build_item_tokens(kwargs)
        h_user, h_item, moe_load_balance_loss = self._forward_sasrec(user_tokens, item_tokens)

        if self.ctr_shallow_shortcut:
            h_shallow = self._shallow_encode(user_tokens)
            ctr_input = torch.cat([h_item, h_shallow], dim=-1)
        else:
            ctr_input = h_item

        logits_click = self.ctr_tower(ctr_input).squeeze(-1)
        gen_logits = self.gen_head(h_user)

        if labels_click is None:
            return torch.sigmoid(logits_click)

        click_targets = labels_click.float().view(-1)
        ctr_loss = F.binary_cross_entropy_with_logits(logits_click, click_targets)

        gen_loss = torch.tensor(0.0, device=logits_click.device)
        if labels_target is not None:
            target_targets = labels_target.long().view(-1)
            gen_loss = F.cross_entropy(gen_logits, target_targets, ignore_index=0)

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


class SASRecBlock(nn.Module):
    """SASRec-style block with mixed attention mask support."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PointWiseFeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-8)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-8)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.attention(self.norm1(x), x, padding_mask, attention_mask)
        x = self.ffn(self.norm2(x), x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with mixed mask (no causal mask)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        padding_value = -1e9
        if padding_mask is not None:
            key_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_mask, padding_value)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(0).unsqueeze(0), padding_value)

        attn_weights = F.softmax(scores, dim=-1)

        if padding_mask is not None:
            query_mask = (~padding_mask).unsqueeze(1).unsqueeze(-1)
            attn_weights = attn_weights * query_mask

        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = out + query
        return out


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network with residual inside."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.dropout(F.relu(self.fc1(x))))
        out = self.dropout(out)
        return out + residual
