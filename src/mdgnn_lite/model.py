from __future__ import annotations

import torch
from torch import nn


def tensor_debug(name: str, value: torch.Tensor) -> str:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = detached.numel()
    if finite_count:
        vals = detached[finite]
        min_val = float(vals.min().item())
        max_val = float(vals.max().item())
        mean_val = float(vals.float().mean().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
        mean_val = float("nan")
    return (
        f"{name}: shape={tuple(detached.shape)} "
        f"finite={finite_count}/{total} min={min_val:.6g} "
        f"max={max_val:.6g} mean={mean_val:.6g}"
    )


class RelationGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weights)

    def forward(self, x: torch.Tensor, relations: torch.Tensor, debug: bool = False) -> torch.Tensor:
        # x: [B, N, D], relations: [R, N, N]
        deg = relations.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_rel = relations / deg
        neigh = torch.einsum("rij,bjd->brid", norm_rel, x)
        out = torch.einsum("brid,rdo->biro", neigh, self.relation_weights).sum(dim=1)
        out = self.dropout(torch.relu(out + self.self_linear(x)))
        if debug:
            print(tensor_debug("graph.deg", deg))
            print(tensor_debug("graph.norm_rel", norm_rel))
            print(tensor_debug("graph.neigh", neigh))
            print(tensor_debug("graph.out", out))
        return out


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

    def forward(self, x: torch.Tensor, relations: torch.Tensor, debug: bool = False) -> torch.Tensor:
        # x: [B, N, T, F]
        batch, num_nodes, window, feat_dim = x.shape
        if debug:
            print(tensor_debug("forward.x", x))
            print(tensor_debug("forward.relations", relations))
        seq = x.reshape(batch * num_nodes, window, feat_dim)
        _, h = self.temporal(seq)
        h = h[-1].reshape(batch, num_nodes, -1)
        if debug:
            print(tensor_debug("forward.temporal_h", h))
        for layer in self.gnn_layers:
            h = layer(h, relations, debug=debug)
        h = self.cross_stock(h)
        pred = self.head(h).squeeze(-1)
        if debug:
            print(tensor_debug("forward.cross_stock", h))
            print(tensor_debug("forward.pred", pred))
        return pred
