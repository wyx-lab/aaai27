from __future__ import annotations

import argparse
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Qlib provider instruments and dataset segments.")
    parser.add_argument("--config", default="configs/master_alpha158_akshare.yaml")
    parser.add_argument("--segments", default="train,valid,test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    qlib_init = config.get("qlib_init", {})
    provider_uri = Path(qlib_init.get("provider_uri", "")).expanduser()
    handler_kwargs = config["dataset"]["kwargs"]["handler"]["kwargs"]
    instruments = handler_kwargs.get("instruments")

    print(f"provider_uri={provider_uri}")
    print(f"configured_instruments={instruments}")
    show_provider_files(provider_uri, str(instruments))

    qlib.init(**qlib_init)
    dataset = init_instance_by_config(config["dataset"])
    for segment in [s.strip() for s in args.segments.split(",") if s.strip()]:
        try:
            df = dataset.prepare(segment, col_set=["feature", "label"], data_key="learn")
        except Exception as exc:
            print(f"segment[{segment}]: ERROR {type(exc).__name__}: {exc}")
            continue
        if df is None or df.empty:
            columns = None if df is None else list(df.columns)[:5]
            print(f"segment[{segment}]: empty shape={None if df is None else df.shape} columns={columns}")
            continue
        dates = df.index.get_level_values("datetime")
        inst = df.index.get_level_values("instrument")
        print(
            f"segment[{segment}]: shape={df.shape} "
            f"dates={dates.min().date()}..{dates.max().date()} "
            f"n_dates={dates.nunique()} n_instruments={inst.nunique()}"
        )


def show_provider_files(provider_uri: Path, instruments: str) -> None:
    inst_path = provider_uri / "instruments" / f"{instruments}.txt"
    print(f"instrument_file={inst_path} exists={inst_path.exists()}")
    if inst_path.exists():
        rows = [line.strip() for line in inst_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"instrument_rows={len(rows)} sample={rows[:5]}")
    feature_dir = provider_uri / "features"
    print(f"features_dir={feature_dir} exists={feature_dir.exists()}")
    if feature_dir.exists():
        samples = sorted(path.name for path in feature_dir.iterdir() if path.is_dir())[:10]
        print(f"feature_symbol_dirs_sample={samples}")


if __name__ == "__main__":
    main()
