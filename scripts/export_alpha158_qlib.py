from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qlib Alpha158 features and labels for a market universe."
    )
    parser.add_argument(
        "--provider-uri",
        default=str(Path.home() / ".qlib" / "qlib_data" / "cn_data"),
        help="Qlib data provider directory, usually ~/.qlib/qlib_data/cn_data.",
    )
    parser.add_argument("--instruments", default="csi300", help="Qlib instrument universe.")
    parser.add_argument("--start", default="2018-01-01", help="Feature start date.")
    parser.add_argument("--end", default="2023-12-31", help="Feature end date.")
    parser.add_argument("--fit-start", default="2018-01-01", help="Normalizer fit start date.")
    parser.add_argument("--fit-end", default="2019-12-31", help="Normalizer fit end date.")
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=2,
        help=(
            "Forward label horizon. Qlib default is 2: Ref($close, -2) / "
            "Ref($close, -1) - 1. Use 5 for five-day-ahead tradable return."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/alpha158",
        help="Output directory for exported parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158

    provider_uri = Path(args.provider_uri).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)

    handler = Alpha158(
        instruments=args.instruments,
        start_time=args.start,
        end_time=args.end,
        fit_start_time=args.fit_start,
        fit_end_time=args.fit_end,
    )

    features = handler.fetch(col_set="feature")
    labels = build_forward_label(
        instruments=args.instruments,
        start=args.start,
        end=args.end,
        horizon=args.label_horizon,
    )

    suffix = f"{args.instruments}_{args.start}_{args.end}".replace("-", "")
    feature_path = out_dir / f"features_alpha158_{suffix}.parquet"
    label_path = out_dir / f"labels_h{args.label_horizon}_{suffix}.parquet"

    features.to_parquet(feature_path)
    labels.to_parquet(label_path)

    print(f"features: {feature_path.resolve()} shape={features.shape}")
    print(f"labels:   {label_path.resolve()} shape={labels.shape}")


def build_forward_label(instruments: str, start: str, end: str, horizon: int):
    if horizon < 2:
        raise ValueError("label-horizon should be >= 2 for tradable T+1 to T+horizon return")
    from qlib.data import D

    expr = f"Ref($close, -{horizon}) / Ref($close, -1) - 1"
    instruments_config = D.instruments(instruments) if isinstance(instruments, str) else instruments
    labels = D.features(
        instruments=instruments_config,
        fields=[expr],
        start_time=start,
        end_time=end,
        freq="day",
    )
    labels.columns = ["LABEL0"]
    return labels


if __name__ == "__main__":
    main()
