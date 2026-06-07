from __future__ import annotations

import torch
from torch import nn


class RelationGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weights)

    def forward(self, x: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], relations: [R, N, N]
        deg = relations.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_rel = relations / deg
        neigh = torch.einsum("rij,bjd->brid", norm_rel, x)
        out = torch.einsum("brid,rdo->biro", neigh, self.relation_weights).sum(dim=1)
        return self.dropout(torch.relu(out + self.self_linear(x)))


class MDGNNLite(nn.Module):
    """A compact MDGNN-style model.

    Temporal encoder maps Alpha158 windows to stock embeddings, relation GCN mixes
    multi-relation graph information, and a Transformer layer models cross-stock context.
    """

    def __init__(
        self,
        feature_dim: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.gnn_layers = nn.ModuleList(
            RelationGraphConv(hidden_dim, hidden_dim, num_relations, dropout)
            for _ in range(num_gnn_layers)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_stock = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F]
        batch, num_nodes, window, feat_dim = x.shape
        seq = x.reshape(batch * num_nodes, window, feat_dim)
        _, h = self.temporal(seq)
        h = h[-1].reshape(batch, num_nodes, -1)
        for layer in self.gnn_layers:
            h = layer(h, relations)
        h = self.cross_stock(h)
        return self.head(h).squeeze(-1)

