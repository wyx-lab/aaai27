from __future__ import annotations

from collections import OrderedDict


GROUP_ORDER = [
    "kbar",
    "price_ma",
    "volume_ma",
    "price_std",
    "volume_std",
    "price_corr",
    "volume_corr",
    "price_rank",
    "volume_rank",
    "price_quantile",
    "volume_quantile",
    "price_regression",
    "volume_regression",
    "count",
    "other",
]


def build_alpha158_groups(feature_names: list[str]) -> OrderedDict[str, list[int]]:
    groups: OrderedDict[str, list[int]] = OrderedDict((name, []) for name in GROUP_ORDER)
    for idx, name in enumerate(feature_names):
        group = infer_alpha158_group(str(name).upper())
        groups[group].append(idx)
    return OrderedDict((name, indices) for name, indices in groups.items() if indices)


def infer_alpha158_group(name: str) -> str:
    if name in {"KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"}:
        return "kbar"
    if name.startswith("MA"):
        return "volume_ma" if name.startswith("MAV") else "price_ma"
    if name.startswith("STD"):
        return "volume_std" if name.startswith("STDV") else "price_std"
    if name.startswith("CORR"):
        return "volume_corr" if "V" in name else "price_corr"
    if name.startswith("RANK"):
        return "volume_rank" if "V" in name else "price_rank"
    if name.startswith(("QTLU", "QTLD")):
        return "volume_quantile" if "V" in name else "price_quantile"
    if name.startswith(("RESI", "RSQR", "SUMP", "SUMN", "SUMD")):
        return "volume_regression" if "V" in name else "price_regression"
    if name.startswith(("CNTP", "CNTN", "CNTD")):
        return "count"
    if "VOLUME" in name or name.endswith("V") or name.startswith("V"):
        return "volume_ma"
    return "other"
