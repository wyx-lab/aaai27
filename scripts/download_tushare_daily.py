from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download A-share daily bars from TuShare for Qlib conversion.")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--raw-dir", default="data/tushare_raw")
    parser.add_argument("--csv-dir", default="data/tushare_qlib_csv")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Optional stock count limit for smoke tests.")
    parser.add_argument("--token", default=os.getenv("TUSHARE_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise ValueError("Set TUSHARE_TOKEN or pass --token")

    raw_dir = Path(args.raw_dir)
    csv_dir = Path(args.csv_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    ts.set_token(args.token)
    pro = ts.pro_api()

    stock_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,exchange,list_date")
    if args.limit:
        stock_basic = stock_basic.head(args.limit)
    stock_basic.to_csv(raw_dir / "stock_basic.csv", index=False)

    trade_cal = pro.trade_cal(exchange="", start_date=args.start, end_date=args.end)
    trade_cal.to_csv(raw_dir / "trade_cal.csv", index=False)

    for ts_code in tqdm(stock_basic["ts_code"].tolist(), desc="stocks"):
        qlib_symbol = to_qlib_symbol(ts_code)
        out_path = csv_dir / f"{qlib_symbol}.csv"
        if out_path.exists():
            continue

        daily = pro.daily(ts_code=ts_code, start_date=args.start, end_date=args.end)
        time.sleep(args.sleep)
        adj = pro.adj_factor(ts_code=ts_code, start_date=args.start, end_date=args.end)
        time.sleep(args.sleep)
        if daily.empty:
            continue

        df = daily.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
        df = df.sort_values("trade_date")
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        df["symbol"] = qlib_symbol
        df["factor"] = df["adj_factor"].ffill().bfill().fillna(1.0)
        df = df.rename(columns={"vol": "volume"})
        qlib_df = df[["date", "symbol", "open", "high", "low", "close", "volume", "factor"]]
        qlib_df.to_csv(out_path, index=False)

    write_instruments(stock_basic, csv_dir, args.start, args.end)
    print(f"raw_dir={raw_dir.resolve()}")
    print(f"csv_dir={csv_dir.resolve()}")


def to_qlib_symbol(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    prefix = exchange.lower()
    return f"{prefix}{code}"


def write_instruments(stock_basic: pd.DataFrame, csv_dir: Path, start: str, end: str) -> None:
    inst_dir = csv_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end, format="%Y%m%d").strftime("%Y-%m-%d")
    rows = []
    for row in stock_basic.itertuples(index=False):
        symbol = to_qlib_symbol(row.ts_code)
        list_date = pd.to_datetime(str(row.list_date), format="%Y%m%d", errors="coerce")
        begin = max(start_date, list_date.strftime("%Y-%m-%d") if pd.notna(list_date) else start_date)
        rows.append(f"{symbol}\t{begin}\t{end_date}\n")
    (inst_dir / "all.txt").write_text("".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
