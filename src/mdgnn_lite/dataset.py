from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Alpha158Meta:
    dates: list[pd.Timestamp]
    instruments: list[str]
    feature_dim: int
    feature_nan_count: int
    label_nan_count: int
    valid_ratio: float


class Alpha158WindowDataset(Dataset):
    """Torch Dataset for Qlib Alpha158 parquet exports.

    Each item is one trading day sample:
      x: [num_stocks, window, feature_dim]
      y: [num_stocks]
      mask: [num_stocks], valid labels/features for loss and metrics
    """

    def __init__(
        self,
        feature_path: str | Path,
        label_path: str | Path,
        start: str | None = None,
        end: str | None = None,
        window: int = 10,
        label_col: str | None = None,
        fillna: float = 0.0,
    ) -> None:
        self.window = window
        self.fillna = fillna

        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path)
        features = _normalize_qlib_frame(features)
        labels = _normalize_qlib_frame(labels)

        if label_col is None:
            label_col = labels.columns[0]

        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index].sort_index()
        labels = labels.loc[common_index, [label_col]].sort_index()

        if start is not None:
            features = features.loc[pd.IndexSlice[pd.Timestamp(start) :, :], :]
            labels = labels.loc[pd.IndexSlice[pd.Timestamp(start) :, :], :]
        if end is not None:
            features = features.loc[pd.IndexSlice[: pd.Timestamp(end), :], :]
            labels = labels.loc[pd.IndexSlice[: pd.Timestamp(end), :], :]

        dates = sorted(features.index.get_level_values("datetime").unique())
        instruments = sorted(features.index.get_level_values("instrument").unique())
        if len(dates) <= window:
            raise ValueError(f"Need more than window={window} dates; got {len(dates)}")

        feature_panel = features.unstack("instrument").reindex(index=dates)
        label_panel = labels[label_col].unstack("instrument").reindex(index=dates, columns=instruments)

        # unstack gives columns as (feature, instrument); transpose to date, instrument, feature.
        arr = feature_panel.to_numpy(dtype=np.float32).reshape(len(dates), -1, len(instruments))
        arr = np.transpose(arr, (0, 2, 1))
        label_arr = label_panel.to_numpy(dtype=np.float32)

        feature_nan_count = int((~np.isfinite(arr)).sum())
        label_nan_count = int((~np.isfinite(label_arr)).sum())
        feature_valid = np.isfinite(arr).all(axis=-1)
        label_valid = np.isfinite(label_arr)
        self.features = np.nan_to_num(arr, nan=fillna, posinf=fillna, neginf=fillna)
        self.labels = np.nan_to_num(label_arr, nan=fillna, posinf=fillna, neginf=fillna)
        self.valid = feature_valid & label_valid
        self.meta = Alpha158Meta(
            dates=dates,
            instruments=instruments,
            feature_dim=self.features.shape[-1],
            feature_nan_count=feature_nan_count,
            label_nan_count=label_nan_count,
            valid_ratio=float(self.valid.mean()),
        )

    def __len__(self) -> int:
        return len(self.meta.dates) - self.window

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = idx + self.window
        x = self.features[idx:end]
        y = self.labels[end]
        mask = self.valid[idx:end].all(axis=0) & self.valid[end]
        return {
            "x": torch.from_numpy(np.transpose(x, (1, 0, 2))).float(),
            "y": torch.from_numpy(np.nan_to_num(y, nan=0.0)).float(),
            "mask": torch.from_numpy(mask).bool(),
        }


def _normalize_qlib_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex dataframe with datetime and instrument levels")
    names = list(df.index.names)
    if "datetime" not in names or "instrument" not in names:
        if len(names) >= 2:
            df.index = df.index.set_names(["datetime", "instrument"] + names[2:])
        else:
            raise ValueError(f"Unexpected index names: {names}")
    return df.sort_index()
