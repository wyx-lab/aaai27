from __future__ import annotations

import torch
from torch import nn


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
        market_dim: int | None = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_beta: float = 5.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.market_dim = market_dim or feature_dim
        self.gate = Gate(feature_dim, self.market_dim, beta=gate_beta)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
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
        h = self.input_proj(x.reshape(batch * num_nodes, window, feat_dim))
        h = self.temporal_attention(h)
        h = self.temporal_agg(h).reshape(batch, num_nodes, -1)
        h = self.spatial_attention(h)
        return self.head(h).squeeze(-1)
