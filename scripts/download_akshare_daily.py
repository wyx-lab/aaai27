from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download A-share daily bars from AKShare for Qlib conversion.")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--raw-dir", default="data/akshare_raw")
    parser.add_argument("--csv-dir", default="data/akshare_qlib_csv")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="Optional stock count limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    csv_dir = Path(args.csv_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    stock_info = ak.stock_info_a_code_name()
    stock_info = normalize_stock_list(stock_info)
    if args.limit:
        stock_info = stock_info.head(args.limit)
    stock_info.to_csv(raw_dir / "stock_info.csv", index=False)

    for row in tqdm(list(stock_info.itertuples(index=False)), desc="stocks"):
        symbol6 = str(row.code).zfill(6)
        qlib_symbol = to_qlib_symbol(symbol6)
        out_path = csv_dir / f"{qlib_symbol}.csv"
        if out_path.exists():
            continue
        try:
            daily = ak.stock_zh_a_hist(
                symbol=symbol6,
                period="daily",
                start_date=args.start,
                end_date=args.end,
                adjust=args.adjust,
            )
        except Exception as exc:
            print(f"skip {symbol6}: {exc}")
            continue
        time.sleep(args.sleep)
        if daily is None or daily.empty:
            continue
        qlib_df = normalize_daily(daily, qlib_symbol)
        if qlib_df.empty:
            continue
        qlib_df.to_csv(out_path, index=False)

    write_instruments(stock_info, csv_dir, args.start, args.end)
    print(f"raw_dir={raw_dir.resolve()}")
    print(f"csv_dir={csv_dir.resolve()}")


def normalize_stock_list(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col in {"code", "代码"}:
            rename[col] = "code"
        elif col in {"name", "名称"}:
            rename[col] = "name"
    df = df.rename(columns=rename)
    if "code" not in df.columns:
        raise ValueError(f"Cannot find stock code column in {list(df.columns)}")
    if "name" not in df.columns:
        df["name"] = ""
    df = df[["code", "name"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def normalize_daily(df: pd.DataFrame, qlib_symbol: str) -> pd.DataFrame:
    col_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; got {list(df.columns)}")
    out = df[required].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["symbol"] = qlib_symbol
    out["factor"] = 1.0
    out = out[["date", "symbol", "open", "high", "low", "close", "volume", "factor"]]
    return out.sort_values("date")


def to_qlib_symbol(symbol6: str) -> str:
    if symbol6.startswith(("5", "6", "9")):
        return f"sh{symbol6}"
    return f"sz{symbol6}"


def write_instruments(stock_info: pd.DataFrame, csv_dir: Path, start: str, end: str) -> None:
    inst_dir = csv_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end, format="%Y%m%d").strftime("%Y-%m-%d")
    rows = [f"{to_qlib_symbol(str(row.code).zfill(6))}\t{start_date}\t{end_date}\n" for row in stock_info.itertuples(index=False)]
    (inst_dir / "all.txt").write_text("".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
