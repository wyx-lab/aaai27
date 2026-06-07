from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def identity_relation(num_nodes: int) -> torch.Tensor:
    return torch.eye(num_nodes, dtype=torch.float32).unsqueeze(0)


def load_relation_tensor(path: str | Path, instruments: list[str]) -> torch.Tensor:
    """Load relation matrices as [num_relations, N, N].

    Supported formats:
      .npy/.npz: array shaped [R, N, N] or [N, N]
      .csv: edge list with src,dst,relation,weight; relation optional
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        arr = data[data.files[0]]
    elif path.suffix == ".csv":
        arr = _edge_csv_to_tensor(path, instruments)
    else:
        raise ValueError(f"Unsupported relation file: {path}")
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return torch.from_numpy(arr.astype(np.float32))


def _edge_csv_to_tensor(path: Path, instruments: list[str]) -> np.ndarray:
    edges = pd.read_csv(path)
    node_to_idx = {s: i for i, s in enumerate(instruments)}
    if "relation" not in edges.columns:
        edges["relation"] = "default"
    if "weight" not in edges.columns:
        edges["weight"] = 1.0
    relations = sorted(edges["relation"].astype(str).unique())
    rel_to_idx = {r: i for i, r in enumerate(relations)}
    arr = np.zeros((len(relations), len(instruments), len(instruments)), dtype=np.float32)
    for row in edges.itertuples(index=False):
        src = getattr(row, "src")
        dst = getattr(row, "dst")
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        arr[rel_to_idx[str(getattr(row, "relation"))], node_to_idx[src], node_to_idx[dst]] = float(
            getattr(row, "weight")
        )
    return arr

