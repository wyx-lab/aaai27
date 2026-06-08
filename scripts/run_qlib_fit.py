from __future__ import annotations

import argparse
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qlib dataset + MDGNN model without workflow recorder.")
    parser.add_argument("--config", default="configs/mdgnn_alpha158.yaml")
    parser.add_argument("--out", default="predictions/mdgnn_scores.parquet")
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


if __name__ == "__main__":
    main()
