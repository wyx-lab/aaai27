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
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--universe", default="csi500", choices=["all", "csi300", "csi500"])
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=0.5)
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
    stock_info = filter_stock_universe(stock_info, args.universe)
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
            daily = fetch_daily_with_retry(
                symbol=qlib_symbol,
                start=args.start,
                end=args.end,
                adjust=args.adjust,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
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
    print(f"universe={args.universe} stocks={len(stock_info)}")


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


def filter_stock_universe(stock_info: pd.DataFrame, universe: str) -> pd.DataFrame:
    if universe == "all":
        return stock_info
    index_code = {"csi300": "000300", "csi500": "000905"}[universe]
    members = {symbol[-6:] for symbol in fetch_index_members(index_code)}
    filtered = stock_info[stock_info["code"].isin(members)].copy()
    if filtered.empty:
        raise ValueError(f"Universe {universe} produced 0 stocks. Check AKShare index constituent API.")
    return filtered


def fetch_daily_with_retry(
    symbol: str,
    start: str,
    end: str,
    adjust: str,
    retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        except Exception as exc:
            last_error = exc
            wait = retry_sleep * attempt
            print(f"retry {attempt}/{retries} {symbol}: {exc}; sleep={wait:.1f}s")
            time.sleep(wait)
    assert last_error is not None
    raise last_error


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
    all_symbols = [to_qlib_symbol(str(row.code).zfill(6)) for row in stock_info.itertuples(index=False)]
    write_instrument_file(inst_dir / "all.txt", all_symbols, start_date, end_date)

    available = set(all_symbols)
    for name, index_code in {"csi300": "000300", "csi500": "000905"}.items():
        try:
            members = fetch_index_members(index_code)
        except Exception as exc:
            print(f"skip instruments/{name}.txt: {exc}")
            continue
        members = [symbol for symbol in members if symbol in available]
        write_instrument_file(inst_dir / f"{name}.txt", members, start_date, end_date)
        print(f"instruments/{name}.txt symbols={len(members)}")


def write_instrument_file(path: Path, symbols: list[str], start_date: str, end_date: str) -> None:
    rows = [f"{symbol}\t{start_date}\t{end_date}\n" for symbol in sorted(set(symbols))]
    path.write_text("".join(rows), encoding="utf-8")


def fetch_index_members(index_code: str) -> list[str]:
    fetchers = [
        lambda: ak.index_stock_cons(symbol=index_code),
        lambda: ak.index_stock_cons_csindex(symbol=index_code),
    ]
    last_error: Exception | None = None
    for fetcher in fetchers:
        try:
            df = fetcher()
            members = normalize_index_members(df)
            if members:
                return members
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"No index members returned for {index_code}")


def normalize_index_members(df: pd.DataFrame) -> list[str]:
    code_col = None
    for col in df.columns:
        lower = str(col).lower()
        if str(col) in {"品种代码", "成分券代码", "证券代码", "代码", "code", "con_code"} or "code" in lower:
            code_col = col
            break
    if code_col is None:
        raise ValueError(f"Cannot find index member code column in {list(df.columns)}")
    codes = df[code_col].astype(str).str.extract(r"(\d{6})", expand=False).dropna().unique()
    return [to_qlib_symbol(code) for code in codes]


if __name__ == "__main__":
    main()
