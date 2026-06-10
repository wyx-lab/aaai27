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
    downloaded = {path.stem.lower() for path in csv_dir.glob("*.csv") if path.stem.lower() not in {"stock_info"}}
    all_symbols = [
        to_qlib_symbol(str(row.code).zfill(6))
        for row in stock_info.itertuples(index=False)
        if to_qlib_symbol(str(row.code).zfill(6)) in downloaded
    ]
    write_instrument_file(inst_dir / "all.txt", all_symbols, start_date, end_date)
    print(f"instruments/all.txt symbols={len(all_symbols)}")

    available = set(all_symbols)
    for name, index_code in {"csi300": "000300", "csi500": "000905"}.items():
        members = [symbol for symbol in fetch_index_members(index_code) if symbol in available]
        write_instrument_file(inst_dir / f"{name}.txt", members, start_date, end_date)
        print(f"instruments/{name}.txt symbols={len(members)}")


if __name__ == "__main__":
    main()
