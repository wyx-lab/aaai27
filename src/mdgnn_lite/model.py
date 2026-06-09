from __future__ import annotations

import math

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[-2]].unsqueeze(0)


class FeatureGate(nn.Module):
    def __init__(self, feature_dim: int, beta: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F]. Use latest features as a market-free proxy for feature selection.
        gate_input = x[:, :, -1, :]
        gate = torch.softmax(self.proj(gate_input) / self.beta, dim=-1) * self.feature_dim
        return x * gate.unsqueeze(2)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal mode: [B*N, T, D]. Spatial mode: [B, N, D].
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        z = self.norm2(x)
        return x + self.ffn(z)


class TemporalPoolingAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, D]. Query with the latest hidden state.
        h = self.proj(x)
        query = h[:, -1:, :].transpose(1, 2)
        weights = torch.softmax(torch.matmul(h, query).squeeze(-1), dim=1).unsqueeze(1)
        return torch.matmul(weights, x).squeeze(1)


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
        use_master_attention: bool = False,
        feature_gate: bool = False,
        gate_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_master_attention = use_master_attention
        self.feature_gate = FeatureGate(feature_dim, gate_beta) if feature_gate else None
        if use_master_attention:
            self.input_proj = nn.Linear(feature_dim, hidden_dim)
            self.position = PositionalEncoding(hidden_dim)
            self.temporal_attention = AttentionBlock(hidden_dim, num_heads, dropout)
            self.temporal_pool = TemporalPoolingAttention(hidden_dim)
            self.spatial_attention = AttentionBlock(hidden_dim, num_heads, dropout)
        else:
            self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_stock = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.gnn_layers = nn.ModuleList(
            RelationGraphConv(hidden_dim, hidden_dim, num_relations, dropout)
            for _ in range(num_gnn_layers)
        )
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
        if self.feature_gate is not None:
            x = self.feature_gate(x)
            if debug:
                print(tensor_debug("forward.feature_gated_x", x))
        seq = x.reshape(batch * num_nodes, window, feat_dim)
        if self.use_master_attention:
            seq = self.position(self.input_proj(seq))
            seq = self.temporal_attention(seq)
            h = self.temporal_pool(seq).reshape(batch, num_nodes, -1)
        else:
            _, h = self.temporal(seq)
            h = h[-1].reshape(batch, num_nodes, -1)
        if debug:
            print(tensor_debug("forward.temporal_h", h))
        for layer in self.gnn_layers:
            h = layer(h, relations, debug=debug)
        h = self.spatial_attention(h) if self.use_master_attention else self.cross_stock(h)
        pred = self.head(h).squeeze(-1)
        if debug:
            print(tensor_debug("forward.cross_stock", h))
            print(tensor_debug("forward.pred", pred))
        return pred
