from __future__ import annotations

import argparse
from pathlib import Path

import akshare as ak
import pandas as pd

from download_akshare_daily import (
    fetch_index_members,
    normalize_stock_list,
    to_qlib_symbol,
    write_instrument_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write all/csi300/csi500 Qlib instrument files for AKShare CSV data.")
    parser.add_argument("--csv-dir", default="data/akshare_qlib_csv")
    parser.add_argument("--qlib-dir", default=None, help="Optional dumped Qlib provider dir to receive instrument files.")
    parser.add_argument("--universe", default="csi500", choices=["all", "csi300", "csi500"])
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    inst_dir = csv_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.to_datetime(args.start, format="%Y%m%d").strftime("%Y-%m-%d")
    end_date = pd.to_datetime(args.end, format="%Y%m%d").strftime("%Y-%m-%d")

    stock_info = normalize_stock_list(ak.stock_info_a_code_name())
    stock_symbols = [to_qlib_symbol(str(row.code).zfill(6)) for row in stock_info.itertuples(index=False)]
    available, source = find_available_symbols(csv_dir, Path(args.qlib_dir) if args.qlib_dir else None)
    if available:
        all_symbols = [symbol for symbol in stock_symbols if symbol in available]
    else:
        source = "stock_info_fallback"
        all_symbols = stock_symbols
    available = set(all_symbols)
    if args.universe == "all":
        all_file_symbols = all_symbols
    else:
        all_file_symbols = [symbol for symbol in fetch_index_members({"csi300": "000300", "csi500": "000905"}[args.universe]) if symbol in available]
    write_instrument_file(inst_dir / "all.txt", all_file_symbols, start_date, end_date)
    print(f"instruments/all.txt symbols={len(all_file_symbols)} source={source} universe={args.universe}")

    for name, index_code in {"csi300": "000300", "csi500": "000905"}.items():
        members = [symbol for symbol in fetch_index_members(index_code) if symbol in available]
        write_instrument_file(inst_dir / f"{name}.txt", members, start_date, end_date)
        print(f"instruments/{name}.txt symbols={len(members)}")

    if args.qlib_dir:
        qlib_inst_dir = Path(args.qlib_dir) / "instruments"
        qlib_inst_dir.mkdir(parents=True, exist_ok=True)
        for path in inst_dir.glob("*.txt"):
            target = qlib_inst_dir / path.name
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"copied {path} -> {target}")


def find_available_symbols(csv_dir: Path, qlib_dir: Path | None) -> tuple[set[str], str]:
    csv_symbols = {path.stem.lower() for path in csv_dir.glob("*.csv") if path.stem.lower() not in {"stock_info"}}
    if csv_symbols:
        return csv_symbols, "csv_dir"
    if qlib_dir is not None:
        feature_dir = qlib_dir / "features"
        if feature_dir.exists():
            feature_symbols = {path.name.lower() for path in feature_dir.iterdir() if path.is_dir()}
            if feature_symbols:
                return feature_symbols, "qlib_features"
    return set(), "none"


if __name__ == "__main__":
    main()
