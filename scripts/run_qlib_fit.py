from __future__ import annotations

import argparse
import json
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config

from src.mdgnn_lite.evaluation import evaluate_predictions, extract_label_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qlib dataset + MDGNN model without workflow recorder.")
    parser.add_argument("--config", default="configs/mdgnn_alpha158.yaml")
    parser.add_argument("--out", default="predictions/mdgnn_scores.parquet")
    parser.add_argument("--metrics-out", default="predictions/mdgnn_metrics.json")
    parser.add_argument("--daily-out", default="predictions/mdgnn_daily_metrics.parquet")
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    qlib_init = config.get("qlib_init", {})
    qlib.init(**qlib_init)

    model = init_instance_by_config(config["model"])
    dataset = init_instance_by_config(config["dataset"])
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_frame("score").to_parquet(out)
    print(f"predictions: {out.resolve()} shape={pred.shape}")

    label = extract_label_series(dataset, segment="test", raw=True)
    metrics, daily = evaluate_predictions(pred, label, topk=args.topk)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    daily_out = Path(args.daily_out)
    daily_out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_out)

    print(f"metrics: {metrics_out.resolve()}")
    print(f"daily_metrics: {daily_out.resolve()} shape={daily.shape}")
    print(
        "test_metrics: "
        f"IC={metrics['ic']:.6f} RankIC={metrics['rankic']:.6f} "
        f"TopKRet={metrics['topk_mean_return']:.6f} TopKSharpe={metrics['topk_sharpe']:.6f} "
        f"Precision@K={metrics['precision_at_topk']:.6f} "
        f"LSRet={metrics['long_short_mean_return']:.6f} LSSharpe={metrics['long_short_sharpe']:.6f} "
        f"MDD={metrics['topk_max_drawdown']:.6f} Turnover={metrics['turnover']:.6f} "
        f"days={metrics['n_days']} obs={metrics['n_obs']}"
    )


if __name__ == "__main__":
    main()
