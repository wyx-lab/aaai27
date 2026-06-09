from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioMetrics:
    ic: float
    rankic: float
    topk_mean_return: float
    topk_annual_return: float
    topk_sharpe: float
    topk_max_drawdown: float
    topk_win_rate: float
    precision_at_topk: float
    long_short_mean_return: float
    long_short_annual_return: float
    long_short_sharpe: float
    long_short_max_drawdown: float
    long_short_win_rate: float
    turnover: float
    n_days: int
    n_obs: int


def evaluate_predictions(
    pred: pd.Series,
    label: pd.Series,
    topk: int = 20,
    annual_days: int = 252,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate cross-sectional prediction scores and simple daily portfolios."""
    frame = pd.concat([pred.rename("score"), label.rename("label")], axis=1).dropna()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("No overlapping prediction and label rows for evaluation")

    daily_rows: list[dict[str, float | pd.Timestamp]] = []
    prev_top: set[str] | None = None
    for dt, group in frame.groupby(level="datetime", sort=True):
        group = group.droplevel("datetime")
        group = group.sort_values("score", ascending=False)
        if len(group) < 2:
            continue

        k = min(topk, len(group))
        top = group.head(k)
        bottom = group.tail(k)
        top_set = set(str(x) for x in top.index)
        turnover = np.nan if prev_top is None else 1.0 - len(top_set & prev_top) / max(len(top_set), 1)
        prev_top = top_set

        daily_rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "ic": _corr(group["score"], group["label"], rank=False),
                "rankic": _corr(group["score"], group["label"], rank=True),
                "topk_return": float(top["label"].mean()),
                "precision_at_topk": float((top["label"] > 0).mean()),
                "long_short_return": float(top["label"].mean() - bottom["label"].mean()),
                "turnover": float(turnover) if np.isfinite(turnover) else np.nan,
                "n_stocks": float(len(group)),
            }
        )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("Evaluation has no valid daily groups")
    metrics = PortfolioMetrics(
        ic=_mean(daily["ic"]),
        rankic=_mean(daily["rankic"]),
        topk_mean_return=_mean(daily["topk_return"]),
        topk_annual_return=_annual_return(daily["topk_return"], annual_days),
        topk_sharpe=_sharpe(daily["topk_return"], annual_days),
        topk_max_drawdown=_max_drawdown(daily["topk_return"]),
        topk_win_rate=_win_rate(daily["topk_return"]),
        precision_at_topk=_mean(daily["precision_at_topk"]),
        long_short_mean_return=_mean(daily["long_short_return"]),
        long_short_annual_return=_annual_return(daily["long_short_return"], annual_days),
        long_short_sharpe=_sharpe(daily["long_short_return"], annual_days),
        long_short_max_drawdown=_max_drawdown(daily["long_short_return"]),
        long_short_win_rate=_win_rate(daily["long_short_return"]),
        turnover=_mean(daily["turnover"]),
        n_days=int(len(daily)),
        n_obs=int(len(frame)),
    )
    return metrics.__dict__, daily


def extract_label_series(dataset, segment: str = "test", raw: bool = True) -> pd.Series:
    data_key = "raw" if raw else "learn"
    try:
        df = dataset.prepare(segment, col_set=["label"], data_key=data_key)
    except Exception:
        if raw:
            df = dataset.prepare(segment, col_set=["label"], data_key="learn")
        else:
            raise
    if df is None or df.empty:
        raise ValueError(f"Qlib segment '{segment}' has no labels for evaluation")
    if isinstance(df.columns, pd.MultiIndex) and "label" in df.columns.get_level_values(0):
        label_df = df["label"]
    else:
        label_df = df
    return label_df.iloc[:, 0].rename("label")


def _corr(score: pd.Series, label: pd.Series, rank: bool) -> float:
    if rank:
        score = score.rank(method="average")
        label = label.rank(method="average")
    value = score.corr(label)
    return float(value) if value == value else float("nan")


def _mean(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.mean()) if len(values) else float("nan")


def _annual_return(values: pd.Series, annual_days: int) -> float:
    values = values.dropna()
    if len(values) == 0:
        return float("nan")
    wealth = (1.0 + values).clip(lower=1e-6).prod()
    return float(wealth ** (annual_days / len(values)) - 1.0)


def _sharpe(values: pd.Series, annual_days: int) -> float:
    values = values.dropna()
    std = values.std(ddof=1)
    if len(values) < 2 or not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(values.mean() / std * np.sqrt(annual_days))


def _max_drawdown(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) == 0:
        return float("nan")
    wealth = (1.0 + values).clip(lower=1e-6).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _win_rate(values: pd.Series) -> float:
    values = values.dropna()
    return float((values > 0).mean()) if len(values) else float("nan")
