import torch
import math
from copy import deepcopy
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encoding = self.calculate_positional_encoding().to(device)

    def calculate_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.hidden_size)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0) / self.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x


class SelfAttentionPooling1(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        q = self.Q(x)  # (batch_size, seq_len, hidden_size)
        k = self.K(x)  # (batch_size, seq_len, hidden_size)
        v = self.V(x)  # (batch_size, seq_len, hidden_size)
        scores = torch.matmul(q, k.transpose(-2, -1))  # k becomes (batch_size, hidden_size, seq_length)
        scores = scores / (self.hidden_size ** 0.5)  # (batch_size, seq_len, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        weighted_values = torch.matmul(attn_weights, v)  # (batch_size, seq_len, hidden_size)
        weighted_values = self.dropout(weighted_values)
        output = weighted_values.mean(dim=1)
        return output


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.pos_encoding = PositionalEncoding(hidden_size)

        # Attention pooling to get sequence-level representation
        #self.pooling_score = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, hidden_size)
        mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
        """
        residual = x
        x = self.pos_encoding(x)
        x = self.norm1(x)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_size)
        if mask is not None:
            mask_2d = mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask_2d == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.bmm(attn_weights, V)  # (batch_size, seq_len, hidden_size)
        x = self.out_proj(context)
        x = self.dropout(x)
        x = self.norm2(x + residual)  # Residual connection

        # Attention pooling
        #scores = self.pooling_score(x).squeeze(-1)  # (batch_size, seq_len)
        #if mask is not None:
        #    scores = scores.masked_fill(mask == 0, -1e9)
        #weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len)
        #output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, hidden_size)

        # Mean pooling
        if mask is not None:
            mask_2d = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            x = x * mask_2d
            output = x.sum(dim=1) / mask_2d.sum(dim=1).clamp(min=1e-9)
        else:
            output = x.mean(dim=1)

        return output


class MultiheadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, dropout=0.2, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size

        self.pos_encoding = PositionalEncoding(hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Attention pooling to get sequence-level representation
        #self.pooling_score = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, hidden_size)
        mask: (batch_size, seq_len) - bool tensor, True for valid, False for pad
        """
        residual = x
        x = self.pos_encoding(x)
        x = self.norm1(x)

        # attention_mask: True means MASKED, so need to inverse the input mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # shape: (batch_size, seq_len)

        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)  # shape: (B, L, H)
        x = self.dropout(attn_output)
        x = self.norm2(x + residual)

        # Attention pooling
        #scores = self.pooling_score(x).squeeze(-1)  # (B, L)
        #if mask is not None:
        #    scores = scores.masked_fill(~mask.bool(), -1e9)
        #weights = torch.softmax(scores, dim=-1)
        #output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, H)

        # Mean pooling
        if mask is not None:
            mask_2d = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            x = x * mask_2d
            output = x.sum(dim=1) / mask_2d.sum(dim=1).clamp(min=1e-9)
        else:
            output = x.mean(dim=1)

        return output
