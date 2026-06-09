from __future__ import annotations

import argparse
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the actual Alpha158 feature columns produced by Qlib.")
    parser.add_argument("--config", default="configs/mdgnn_alpha158_akshare.yaml")
    parser.add_argument("--segment", default="train")
    parser.add_argument("--rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    qlib.init(**config.get("qlib_init", {}))
    dataset = init_instance_by_config(config["dataset"])
    df = dataset.prepare(args.segment, col_set=["feature", "label"], data_key="learn")

    if df is None or df.empty:
        print(f"segment={args.segment} is empty")
        return

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1 and "feature" in df.columns.get_level_values(0):
        features = df["feature"]
        labels = df["label"] if "label" in df.columns.get_level_values(0) else None
    else:
        features = df
        labels = None

    dates = df.index.get_level_values("datetime")
    instruments = df.index.get_level_values("instrument")
    print(f"config={Path(args.config).resolve()}")
    print(f"segment={args.segment} shape={df.shape}")
    print(f"dates={dates.min().date()}..{dates.max().date()} n_dates={dates.nunique()}")
    print(f"instruments={instruments.nunique()}")
    print(f"feature_dim={features.shape[1]}")
    print(f"label_dim={0 if labels is None else labels.shape[1]}")
    print(f"first_features={list(features.columns[: min(args.rows, features.shape[1])])}")


if __name__ == "__main__":
    main()
