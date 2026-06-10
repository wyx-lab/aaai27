from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from qlib.model.base import Model

from .alpha_groups import build_alpha158_groups
from .master_model import MASTERLite
from .qlib_model import PanelMeta, QlibPanelDataset, print_frame_info, split_frame_by_dates


class MASTERQlibModel(Model):
    def __init__(
        self,
        window: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_beta: float = 5.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        epochs: int = 20,
        device: str | None = None,
        feature_norm: str = "none",
        feature_clip: float | None = None,
        label_norm: str = "none",
        label_clip: float | None = None,
        use_alpha_groups: bool = True,
    ) -> None:
        self.window = window
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.gate_beta = gate_beta
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_norm = feature_norm
        self.feature_clip = feature_clip
        self.label_norm = label_norm
        self.label_clip = label_clip
        self.use_alpha_groups = use_alpha_groups
        self.model: MASTERLite | None = None
        self.feature_center: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.label_center: np.ndarray | None = None
        self.label_scale: np.ndarray | None = None
        self.meta: PanelMeta | None = None

    def fit(self, dataset, evals_result=None, **kwargs):
        train_df = dataset.prepare("train", col_set=["feature", "label"], data_key="learn")
        valid_df = dataset.prepare("valid", col_set=["feature", "label"], data_key="learn")
        if train_df is None or train_df.empty:
            raise ValueError(
                "Qlib segment 'train' is empty. Run "
                "`python scripts/check_qlib_segments.py --config configs/master_alpha158_akshare.yaml` "
                "and verify the configured instruments file exists under the Qlib provider directory."
            )
        if valid_df is None or valid_df.empty:
            print("segment[valid] is empty; splitting tail validation from train segment")
            train_df, valid_df = split_frame_by_dates(train_df, valid_ratio=0.2)
        print_frame_info("train", train_df)
        print_frame_info("valid", valid_df)
        train_x, train_y, train_meta = self._frame_to_panel(train_df, segment="train")
        valid_x, valid_y, valid_meta = self._frame_to_panel(valid_df, segment="valid", instruments=train_meta.instruments)
        feature_names = extract_feature_names(train_df)
        group_indices = build_alpha158_groups(feature_names) if self.use_alpha_groups else None
        if group_indices:
            print("alpha_groups: " + ", ".join(f"{name}={len(indices)}" for name, indices in group_indices.items()))
        self._fit_feature_norm(train_x)
        self._fit_label_norm(train_y)
        train_x = self._transform_features(train_x)
        valid_x = self._transform_features(valid_x)
        train_y = self._transform_labels(train_y)
        valid_y = self._transform_labels(valid_y)
        self.meta = train_meta

        train_loader = DataLoader(QlibPanelDataset(train_x, train_y, self.window), self.batch_size, shuffle=True)
        valid_loader = DataLoader(QlibPanelDataset(valid_x, valid_y, self.window), self.batch_size, shuffle=False)
        self.model = MASTERLite(
            feature_dim=train_meta.feature_dim,
            group_indices=group_indices,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            gate_beta=self.gate_beta,
        ).to(self.device)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        print(
            "qlib_master: "
            f"train_dates={train_meta.dates[0].date()}..{train_meta.dates[-1].date()} "
            f"valid_dates={valid_meta.dates[0].date()}..{valid_meta.dates[-1].date()} "
            f"instruments={len(train_meta.instruments)} feature_dim={train_meta.feature_dim} "
            f"window={self.window} batch_size={self.batch_size} "
            f"feature_norm={self.feature_norm} label_norm={self.label_norm} "
            f"alpha_groups={self.use_alpha_groups}"
        )
        print("loss_fn: MSELoss; score: raw model output")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_count = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = loss_fn(pred, y)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optim.step()
                train_loss += float(loss.detach()) * y.numel()
                train_count += y.numel()
            valid_loss, valid_ic, valid_rankic = self._evaluate(valid_loader, loss_fn)
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
        x, _, meta = self._frame_to_panel(test_df, has_label=False, segment=segment, instruments=self.meta.instruments)
        x = self._transform_features(x)
        loader = DataLoader(QlibPanelDataset(x, np.zeros((len(x), len(meta.instruments)), dtype=np.float32), self.window), 1)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                pred = self.model(xb.to(self.device))
                preds.append(pred.detach().cpu().numpy()[0])
        index = pd.MultiIndex.from_product([meta.dates[self.window :], meta.instruments], names=["datetime", "instrument"])
        return pd.Series(np.concatenate(preds), index=index, name="score")

    def _frame_to_panel(self, df: pd.DataFrame, has_label: bool = True, segment: str = "unknown", instruments: list[str] | None = None):
        if df is None or df.empty:
            raise ValueError(f"Qlib segment '{segment}' is empty after prepare().")
        df = df.sort_index()
        if isinstance(df.columns, pd.MultiIndex):
            top = df.columns.get_level_values(0)
            feature_df = df["feature"] if "feature" in top else df
            label_df = df["label"] if has_label and "label" in top else None
        else:
            feature_df = df
            label_df = None
        dates = sorted(df.index.get_level_values("datetime").unique())
        observed_instruments = sorted(df.index.get_level_values("instrument").unique())
        instruments = instruments or observed_instruments
        feature_panel = feature_df.unstack("instrument").reindex(index=dates)
        feature_panel = feature_panel.reindex(columns=pd.MultiIndex.from_product([feature_df.columns, instruments]))
        arr = feature_panel.to_numpy(dtype=np.float32).reshape(len(dates), -1, len(instruments))
        arr = np.transpose(arr, (0, 2, 1))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if label_df is not None:
            label_panel = label_df.iloc[:, 0].unstack("instrument").reindex(index=dates, columns=instruments)
            labels = np.nan_to_num(label_panel.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            labels = np.zeros((len(dates), len(instruments)), dtype=np.float32)
        return arr, labels, PanelMeta(dates=dates, instruments=instruments, feature_dim=arr.shape[-1])

    def _fit_feature_norm(self, x: np.ndarray) -> None:
        if self.feature_norm == "none":
            self.feature_center = None
            self.feature_scale = None
        elif self.feature_norm == "ts_zscore":
            self.feature_center = x.mean(axis=0, keepdims=True)
            self.feature_scale = x.std(axis=0, keepdims=True)
        elif self.feature_norm == "zscore":
            self.feature_center = x.mean(axis=(0, 1), keepdims=True)
            self.feature_scale = x.std(axis=(0, 1), keepdims=True)
        elif self.feature_norm == "robust":
            self.feature_center = np.median(x, axis=(0, 1), keepdims=True)
            self.feature_scale = np.percentile(x, 75, axis=(0, 1), keepdims=True) - np.percentile(
                x, 25, axis=(0, 1), keepdims=True
            )
        else:
            raise ValueError(f"Unsupported feature_norm={self.feature_norm}")

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_center is not None and self.feature_scale is not None:
            x = (x - self.feature_center) / np.maximum(self.feature_scale, 1e-6)
        if self.feature_clip is not None:
            x = np.clip(x, -self.feature_clip, self.feature_clip)
        return x.astype(np.float32)

    def _fit_label_norm(self, y: np.ndarray) -> None:
        if self.label_norm == "none":
            self.label_center = None
            self.label_scale = None
        elif self.label_norm == "robust":
            self.label_center = np.median(y, keepdims=True)
            self.label_scale = np.percentile(y, 75, keepdims=True) - np.percentile(y, 25, keepdims=True)
        elif self.label_norm == "zscore":
            self.label_center = y.mean(keepdims=True)
            self.label_scale = y.std(keepdims=True)
        else:
            raise ValueError(f"Unsupported label_norm={self.label_norm}")

    def _transform_labels(self, y: np.ndarray) -> np.ndarray:
        if self.label_center is not None and self.label_scale is not None:
            y = (y - self.label_center) / np.maximum(self.label_scale, 1e-6)
        if self.label_clip is not None:
            y = np.clip(y, -self.label_clip, self.label_clip)
        return y.astype(np.float32)

    def _evaluate(self, loader, loss_fn):
        assert self.model is not None
        self.model.eval()
        losses: list[float] = []
        ics: list[float] = []
        rankics: list[float] = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                losses.append(float(loss_fn(pred, y).detach()))
                ics.extend(_batch_corr(pred, y, rank=False))
                rankics.extend(_batch_corr(pred, y, rank=True))
        return _mean(losses), _mean(ics), _mean(rankics)


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


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=x.dtype)
    return ranks


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def extract_feature_names(df: pd.DataFrame) -> list[str]:
    if isinstance(df.columns, pd.MultiIndex) and "feature" in df.columns.get_level_values(0):
        return [str(col) for col in df["feature"].columns]
    return [str(col) for col in df.columns]
