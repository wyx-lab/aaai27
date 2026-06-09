from __future__ import annotations

import argparse
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Alpha158 feature ranges from a Qlib config.")
    parser.add_argument("--config", default="configs/mdgnn_alpha158_akshare.yaml")
    parser.add_argument("--segment", default="train")
    parser.add_argument("--out", default="reports/alpha158_feature_ranges.csv")
    parser.add_argument("--data-key", default="learn", choices=["learn", "infer", "raw"])
    parser.add_argument("--feature-norm", default=None, choices=["config", "none", "zscore", "ts_zscore", "robust"])
    parser.add_argument("--feature-clip", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    qlib.init(**config.get("qlib_init", {}))
    dataset = init_instance_by_config(config["dataset"])
    df = dataset.prepare(args.segment, col_set=["feature"], data_key=args.data_key)
    if df is None or df.empty:
        raise ValueError(f"segment={args.segment} data_key={args.data_key} has no features")

    features = df["feature"] if getattr(df.columns, "nlevels", 1) > 1 and "feature" in df.columns.get_level_values(0) else df
    model_kwargs = config.get("model", {}).get("kwargs", {})
    feature_norm = args.feature_norm or model_kwargs.get("feature_norm", "none")
    if feature_norm == "config":
        feature_norm = model_kwargs.get("feature_norm", "none")
    feature_clip = args.feature_clip if args.feature_clip is not None else model_kwargs.get("feature_clip")
    features = normalize_features(features, feature_norm=feature_norm, feature_clip=feature_clip)
    summary = features.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary = summary.rename(columns={"50%": "median"})
    summary["range"] = summary["max"] - summary["min"]
    summary = summary[
        [
            "count",
            "mean",
            "std",
            "min",
            "1%",
            "5%",
            "median",
            "95%",
            "99%",
            "max",
            "range",
        ]
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out)
    print(
        f"feature_ranges: {out.resolve()} shape={summary.shape} "
        f"feature_norm={feature_norm} feature_clip={feature_clip}"
    )
    print(summary.sort_values("range", ascending=False).head(20).to_string())


def normalize_features(features, feature_norm: str, feature_clip: float | None):
    if feature_norm == "none":
        out = features.copy()
    elif feature_norm == "zscore":
        out = (features - features.mean(axis=0)) / features.std(axis=0).replace(0, 1.0)
    elif feature_norm == "robust":
        q75 = features.quantile(0.75)
        q25 = features.quantile(0.25)
        out = (features - features.median(axis=0)) / (q75 - q25).replace(0, 1.0)
    elif feature_norm == "ts_zscore":
        out = features.groupby(level="instrument", group_keys=False).apply(
            lambda frame: (frame - frame.mean(axis=0)) / frame.std(axis=0).replace(0, 1.0)
        )
    else:
        raise ValueError(f"Unsupported feature_norm={feature_norm}")
    out = out.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    if feature_clip is not None:
        out = out.clip(lower=-feature_clip, upper=feature_clip)
    return out


if __name__ == "__main__":
    main()
