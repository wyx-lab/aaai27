from __future__ import annotations

import torch
from torch import nn


class AlphaGroupEncoder(nn.Module):
    def __init__(self, group_indices: dict[str, list[int]], hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if not group_indices:
            raise ValueError("group_indices must not be empty")
        self.group_names = list(group_indices)
        self.group_indices = {name: torch.tensor(indices, dtype=torch.long) for name, indices in group_indices.items()}
        self.group_proj = nn.ModuleDict(
            {name: nn.Linear(len(indices), hidden_dim) for name, indices in group_indices.items()}
        )
        self.group_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.group_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, F]
        states = []
        for name in self.group_names:
            idx = self.group_indices[name].to(x.device)
            states.append(self.group_proj[name](x.index_select(dim=-1, index=idx)))
        group_states = torch.stack(states, dim=2)  # [B*N, T, G, H]
        batch_time, num_groups, hidden_dim = group_states.shape[0] * group_states.shape[1], group_states.shape[2], group_states.shape[3]
        tokens = group_states.reshape(batch_time, num_groups, hidden_dim)
        query = self.group_token.expand(batch_time, -1, -1)
        pooled, _ = self.group_attention(query, tokens, tokens, need_weights=False)
        return self.dropout(self.norm(pooled.squeeze(1).reshape(group_states.shape[0], group_states.shape[1], hidden_dim)))


class Gate(nn.Module):
    """Market-guided feature selection gate from MASTER-style models."""

    def __init__(self, feature_dim: int, market_dim: int, beta: float = 5.0) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.market_proj = nn.Linear(market_dim, feature_dim, bias=False)
        self.feature_dim = feature_dim
        self.beta = beta

    def forward(self, x: torch.Tensor, market: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F], market: [B, N, T, M]
        gate = torch.softmax((self.feature_proj(x) + self.market_proj(market)) / self.beta, dim=-1)
        return x * gate * self.feature_dim


class TAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, H]
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class TemporalAggregator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, H]. Use latest hidden state to pool temporal states.
        h = self.query(x)
        q = h[:, -1:, :].transpose(1, 2)
        weights = torch.softmax(torch.matmul(h, q).squeeze(-1), dim=1).unsqueeze(1)
        return torch.matmul(weights, x).squeeze(1)


class SAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, H]
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class MASTERLite(nn.Module):
    """MASTER-style stock predictor for panel Alpha features.

    The original MASTER uses market information as a side input for the feature
    gate. This implementation supports explicit market features, and falls back
    to a cross-sectional market proxy computed from stock features when no side
    input is supplied.
    """

    def __init__(
        self,
        feature_dim: int,
        group_indices: dict[str, list[int]] | None = None,
        market_dim: int | None = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_beta: float = 5.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.group_indices = group_indices
        self.market_dim = market_dim or feature_dim
        self.gate = Gate(feature_dim, self.market_dim, beta=gate_beta)
        self.group_encoder = (
            AlphaGroupEncoder(group_indices, hidden_dim, num_heads, dropout) if group_indices else None
        )
        self.input_proj = nn.Linear(feature_dim, hidden_dim) if self.group_encoder is None else None
        self.temporal_attention = TAttention(hidden_dim, num_heads, dropout)
        self.temporal_agg = TemporalAggregator(hidden_dim)
        self.spatial_attention = SAttention(hidden_dim, num_heads, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, market: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, T, F]
        batch, num_nodes, window, feat_dim = x.shape
        if market is None:
            market = x.mean(dim=1, keepdim=True).expand(batch, num_nodes, window, feat_dim)
        x = self.gate(x, market)
        seq = x.reshape(batch * num_nodes, window, feat_dim)
        h = self.group_encoder(seq) if self.group_encoder is not None else self.input_proj(seq)
        h = self.temporal_attention(h)
        h = self.temporal_agg(h).reshape(batch, num_nodes, -1)
        h = self.spatial_attention(h)
        return self.head(h).squeeze(-1)
