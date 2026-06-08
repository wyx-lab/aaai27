from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from qlib.model.base import Model

from .model import MDGNNLite


@dataclass
class PanelMeta:
    dates: list[pd.Timestamp]
    instruments: list[str]
    feature_dim: int


class QlibPanelDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, window: int) -> None:
        self.features = features
        self.labels = labels
        self.window = window

    def __len__(self) -> int:
        return len(self.features) - self.window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = idx + self.window
        x = self.features[idx:end]
        y = self.labels[end]
        return torch.from_numpy(np.transpose(x, (1, 0, 2))).float(), torch.from_numpy(y).float()


class MDGNNQlibModel(Model):
    """Qlib Model wrapper for MDGNN-lite.

    Qlib handles Alpha158 feature construction, processors, segmentation, and
    workflow records. This class only adapts DatasetH prepared data into
    windowed panel tensors and trains/predicts stock scores.
    """

    def __init__(
        self,
        window: int = 10,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        epochs: int = 20,
        device: str | None = None,
        feature_norm: str = "none",
        feature_clip: float | None = None,
        label_clip: float | None = None,
        pos_weight: float | None = None,
    ) -> None:
        self.window = window
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_norm = feature_norm
        self.feature_clip = feature_clip
        self.label_clip = label_clip
        self.pos_weight = pos_weight
        self.model: MDGNNLite | None = None
        self.feature_center: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.meta: PanelMeta | None = None

    def fit(self, dataset, evals_result=None, **kwargs):
        train_df = dataset.prepare("train", col_set=["feature", "label"], data_key="learn")
        valid_df = dataset.prepare("valid", col_set=["feature", "label"], data_key="learn")
        if valid_df is None or valid_df.empty:
            print("segment[valid] is empty; splitting tail validation from train segment")
            train_df, valid_df = split_frame_by_dates(train_df, valid_ratio=0.2)
        print_frame_info("train", train_df)
        print_frame_info("valid", valid_df)
        train_x, train_y, train_meta = self._frame_to_panel(train_df, segment="train")
        valid_x, valid_y, valid_meta = self._frame_to_panel(
            valid_df,
            segment="valid",
            instruments=train_meta.instruments,
        )
        self._fit_feature_norm(train_x)
        train_x = self._transform_features(train_x)
        valid_x = self._transform_features(valid_x)
        train_y = self._transform_labels(train_y)
        valid_y = self._transform_labels(valid_y)
        self.meta = train_meta

        train_loader = DataLoader(QlibPanelDataset(train_x, train_y, self.window), self.batch_size, shuffle=True)
        valid_loader = DataLoader(QlibPanelDataset(valid_x, valid_y, self.window), self.batch_size, shuffle=False)
        self.model = MDGNNLite(
            feature_dim=train_meta.feature_dim,
            num_relations=1,
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)
        relation = torch.eye(len(train_meta.instruments), dtype=torch.float32, device=self.device).unsqueeze(0)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = self._build_wce_loss(train_y)

        print(
            "qlib_mdgnn: "
            f"train_dates={train_meta.dates[0].date()}..{train_meta.dates[-1].date()} "
            f"valid_dates={valid_meta.dates[0].date()}..{valid_meta.dates[-1].date()} "
            f"instruments={len(train_meta.instruments)} feature_dim={train_meta.feature_dim} "
            f"window={self.window} batch_size={self.batch_size} feature_norm={self.feature_norm}"
        )
        print("loss_fn: weighted BCEWithLogitsLoss on sign(label); score: exp(logit); metrics: valid IC, valid RankIC")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_count = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x, relation)
                loss = loss_fn(logit, label_to_binary(y))
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optim.step()
                train_loss += float(loss.detach()) * y.numel()
                train_count += y.numel()
            valid_loss, valid_ic, valid_rankic = self._evaluate(valid_loader, relation, loss_fn)
            print(
                f"epoch={epoch:03d} train_loss={train_loss / max(train_count, 1):.6f} "
                f"valid_loss={valid_loss:.6f} valid_ic={valid_ic:.6f} valid_rankic={valid_rankic:.6f}"
            )

    def predict(self, dataset, segment="test"):
        if self.model is None:
            raise ValueError("Model is not fitted")
        test_df = dataset.prepare(segment, col_set=["feature"], data_key="infer")
        if test_df is None or test_df.empty:
            print(f"segment[{segment}] is empty; falling back to train segment tail for prediction")
            full_df = dataset.prepare("train", col_set=["feature"], data_key="infer")
            _, test_df = split_frame_by_dates(full_df, valid_ratio=0.2)
        print_frame_info(segment, test_df)
        if self.meta is None:
            raise ValueError("Model metadata is missing; fit the model before predict")
        x, _, meta = self._frame_to_panel(
            test_df,
            has_label=False,
            segment=segment,
            instruments=self.meta.instruments,
        )
        x = self._transform_features(x)
        loader = DataLoader(QlibPanelDataset(x, np.zeros((len(x), len(meta.instruments)), dtype=np.float32), self.window), 1)
        relation = torch.eye(len(meta.instruments), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                logit = self.model(xb.to(self.device), relation)
                score = torch.exp(logit.clamp(max=20.0))
                preds.append(score.detach().cpu().numpy()[0])
        index = pd.MultiIndex.from_product(
            [meta.dates[self.window :], meta.instruments],
            names=["datetime", "instrument"],
        )
        return pd.Series(np.concatenate(preds), index=index, name="score")

    def _frame_to_panel(
        self,
        df: pd.DataFrame,
        has_label: bool = True,
        segment: str = "unknown",
        instruments: list[str] | None = None,
    ):
        if df is None or df.empty:
            columns = None if df is None else list(df.columns)
            raise ValueError(
                f"Qlib segment '{segment}' is empty after prepare(). "
                f"Check dataset segments, label config, and processors. columns={columns}"
            )
        df = df.sort_index()
        if isinstance(df.columns, pd.MultiIndex):
            top = df.columns.get_level_values(0)
            if "feature" in top:
                feature_df = df["feature"]
            else:
                feature_df = df
            label_df = df["label"] if has_label and "label" in top else None
        else:
            feature_df = df
            label_df = None
        dates = sorted(df.index.get_level_values("datetime").unique())
        observed_instruments = sorted(df.index.get_level_values("instrument").unique())
        instruments = instruments or observed_instruments
        if not dates or not instruments:
            raise ValueError(
                f"Qlib segment '{segment}' has no dates or instruments after prepare(): "
                f"dates={len(dates)} instruments={len(instruments)} shape={df.shape}"
            )
        feature_panel = feature_df.unstack("instrument").reindex(index=dates)
        feature_panel = feature_panel.reindex(
            columns=pd.MultiIndex.from_product([feature_df.columns, instruments])
        )
        arr = feature_panel.to_numpy(dtype=np.float32).reshape(len(dates), -1, len(instruments))
        arr = np.transpose(arr, (0, 2, 1))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if label_df is not None:
            label_panel = label_df.iloc[:, 0].unstack("instrument").reindex(index=dates, columns=instruments)
            labels = label_panel.to_numpy(dtype=np.float32)
            labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            labels = np.zeros((len(dates), len(instruments)), dtype=np.float32)
        return arr, labels, PanelMeta(dates=dates, instruments=instruments, feature_dim=arr.shape[-1])

    def _fit_feature_norm(self, x: np.ndarray) -> None:
        if self.feature_norm == "none":
            self.feature_center = None
            self.feature_scale = None
            return
        if self.feature_norm == "zscore":
            self.feature_center = x.mean(axis=(0, 1), keepdims=True)
            self.feature_scale = x.std(axis=(0, 1), keepdims=True)
        elif self.feature_norm == "robust":
            self.feature_center = np.median(x, axis=(0, 1), keepdims=True)
            q75 = np.percentile(x, 75, axis=(0, 1), keepdims=True)
            q25 = np.percentile(x, 25, axis=(0, 1), keepdims=True)
            self.feature_scale = q75 - q25
        else:
            raise ValueError(f"Unsupported feature_norm={self.feature_norm}")

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_center is not None and self.feature_scale is not None:
            x = (x - self.feature_center) / np.maximum(self.feature_scale, 1e-6)
        if self.feature_clip is not None:
            x = np.clip(x, -self.feature_clip, self.feature_clip)
        return x.astype(np.float32)

    def _transform_labels(self, y: np.ndarray) -> np.ndarray:
        if self.label_clip is not None:
            y = np.clip(y, -self.label_clip, self.label_clip)
        return y.astype(np.float32)

    def _evaluate(self, loader, relation, loss_fn):
        assert self.model is not None
        self.model.eval()
        losses: list[float] = []
        ics: list[float] = []
        rankics: list[float] = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x, relation)
                score = torch.exp(logit.clamp(max=20.0))
                losses.append(float(loss_fn(logit, label_to_binary(y)).detach()))
                ics.extend(_batch_corr(score, y, rank=False))
                rankics.extend(_batch_corr(score, y, rank=True))
        return _mean(losses), _mean(ics), _mean(rankics)

    def _build_wce_loss(self, train_y: np.ndarray) -> nn.Module:
        if self.pos_weight is not None:
            pos_weight = float(self.pos_weight)
        else:
            target = train_y > 0
            pos = max(float(target.sum()), 1.0)
            neg = max(float(target.size - target.sum()), 1.0)
            pos_weight = neg / pos
        print(f"wce_pos_weight={pos_weight:.6f}")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self.device))


def _batch_corr(pred: torch.Tensor, y: torch.Tensor, rank: bool) -> list[float]:
    values: list[float] = []
    for p, t in zip(pred.detach().cpu().float(), y.detach().cpu().float()):
        if rank:
            p = _rank(p)
            t = _rank(t)
        p = p - p.mean()
        t = t - t.mean()
        denom = torch.sqrt((p.square().sum() * t.square().sum()).clamp_min(1e-12))
        values.append(float((p * t).sum() / denom))
    return values


def label_to_binary(y: torch.Tensor) -> torch.Tensor:
    return (y > 0).float()


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=x.dtype)
    return ranks


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def print_frame_info(name: str, df: pd.DataFrame) -> None:
    if df is None:
        print(f"segment[{name}]: None")
        return
    if df.empty:
        print(f"segment[{name}]: empty shape={df.shape} columns={list(df.columns)}")
        return
    dates = df.index.get_level_values("datetime")
    instruments = df.index.get_level_values("instrument")
    print(
        f"segment[{name}]: shape={df.shape} "
        f"dates={dates.min().date()}..{dates.max().date()} "
        f"n_dates={dates.nunique()} n_instruments={instruments.nunique()} "
        f"columns={list(df.columns)[:5]}"
    )


def split_frame_by_dates(df: pd.DataFrame, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        raise ValueError("Cannot split empty dataframe")
    dates = sorted(df.index.get_level_values("datetime").unique())
    if len(dates) < 3:
        raise ValueError(f"Need at least 3 dates for fallback split, got {len(dates)}")
    valid_size = max(1, int(len(dates) * valid_ratio))
    valid_start = dates[-valid_size]
    train = df[df.index.get_level_values("datetime") < valid_start]
    valid = df[df.index.get_level_values("datetime") >= valid_start]
    if train.empty or valid.empty:
        raise ValueError(
            f"Fallback split failed: train_empty={train.empty} valid_empty={valid.empty} "
            f"n_dates={len(dates)} valid_start={valid_start}"
        )
    return train, valid
