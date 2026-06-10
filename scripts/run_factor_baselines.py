from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.utils import init_instance_by_config

from src.mdgnn_lite.evaluation import evaluate_predictions, extract_label_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flat Alpha158 MLP/XGBoost baselines for factor sanity checks.")
    parser.add_argument("--config", default="configs/mdgnn_alpha158_akshare.yaml")
    parser.add_argument("--model", choices=["mlp", "xgboost"], default="mlp")
    parser.add_argument("--out-dir", default="predictions/baselines")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means use all train rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    qlib.init(**config.get("qlib_init", {}))
    dataset = init_instance_by_config(config["dataset"])
    train = get_feature_label_frame(dataset, "train")
    valid = get_feature_label_frame(dataset, "valid")
    test_x = get_feature_frame(dataset, "test")

    model_kwargs = config.get("model", {}).get("kwargs", {})
    normalizer = FeatureLabelNormalizer(
        feature_norm=model_kwargs.get("feature_norm", "none"),
        feature_clip=model_kwargs.get("feature_clip"),
        label_norm=model_kwargs.get("label_norm", "none"),
        label_clip=model_kwargs.get("label_clip"),
    )
    normalizer.fit(train["feature"], train["label"])
    train_x = normalizer.transform_features(train["feature"])
    train_y = normalizer.transform_labels(train["label"])
    valid_x = normalizer.transform_features(valid["feature"])
    valid_y = normalizer.transform_labels(valid["label"])
    test_x_norm = normalizer.transform_features(test_x)

    if args.max_train_rows and len(train_x) > args.max_train_rows:
        train_x = train_x.sample(args.max_train_rows, random_state=42)
        train_y = train_y.loc[train_x.index]

    print(
        f"baseline={args.model} train={train_x.shape} valid={valid_x.shape} test={test_x_norm.shape} "
        f"feature_norm={normalizer.feature_norm} label_norm={normalizer.label_norm}"
    )

    if args.model == "mlp":
        pred = fit_predict_mlp(args, train_x, train_y, valid_x, valid_y, test_x_norm)
    else:
        pred = fit_predict_xgboost(train_x, train_y, valid_x, valid_y, test_x_norm)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{args.model}_scores.parquet"
    metrics_path = out_dir / f"{args.model}_metrics.json"
    daily_path = out_dir / f"{args.model}_daily_metrics.parquet"

    pred.to_frame("score").to_parquet(pred_path)
    raw_label = extract_label_series(dataset, segment="test", raw=True)
    metrics, daily = evaluate_predictions(pred, raw_label, topk=args.topk)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    daily.to_parquet(daily_path)

    print(f"predictions: {pred_path.resolve()} shape={pred.shape}")
    print(f"metrics: {metrics_path.resolve()}")
    print(
        "test_metrics: "
        f"IC={metrics['ic']:.6f} RankIC={metrics['rankic']:.6f} "
        f"TopKRet={metrics['topk_mean_return']:.6f} TopKSharpe={metrics['topk_sharpe']:.6f} "
        f"Precision@K={metrics['precision_at_topk']:.6f} "
        f"LSRet={metrics['long_short_mean_return']:.6f} LSSharpe={metrics['long_short_sharpe']:.6f} "
        f"days={metrics['n_days']} obs={metrics['n_obs']}"
    )


class FeatureLabelNormalizer:
    def __init__(
        self,
        feature_norm: str = "none",
        feature_clip: float | None = None,
        label_norm: str = "none",
        label_clip: float | None = None,
    ) -> None:
        self.feature_norm = feature_norm
        self.feature_clip = feature_clip
        self.label_norm = label_norm
        self.label_clip = label_clip
        self.feature_center: pd.DataFrame | pd.Series | None = None
        self.feature_scale: pd.DataFrame | pd.Series | None = None
        self.label_center: float | None = None
        self.label_scale: float | None = None

    def fit(self, features: pd.DataFrame, label: pd.Series) -> None:
        if self.feature_norm == "zscore":
            self.feature_center = features.mean(axis=0)
            self.feature_scale = features.std(axis=0).replace(0, 1.0)
        elif self.feature_norm == "robust":
            q75 = features.quantile(0.75)
            q25 = features.quantile(0.25)
            self.feature_center = features.median(axis=0)
            self.feature_scale = (q75 - q25).replace(0, 1.0)
        elif self.feature_norm == "ts_zscore":
            grouped = features.groupby(level="instrument")
            self.feature_center = grouped.mean()
            self.feature_scale = grouped.std().replace(0, 1.0)
        elif self.feature_norm != "none":
            raise ValueError(f"Unsupported feature_norm={self.feature_norm}")

        if self.label_norm == "zscore":
            self.label_center = float(label.mean())
            self.label_scale = float(label.std())
        elif self.label_norm == "robust":
            self.label_center = float(label.median())
            self.label_scale = float(label.quantile(0.75) - label.quantile(0.25))
        elif self.label_norm != "none":
            raise ValueError(f"Unsupported label_norm={self.label_norm}")

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        if self.feature_norm in {"zscore", "robust"}:
            assert isinstance(self.feature_center, pd.Series)
            assert isinstance(self.feature_scale, pd.Series)
            out = (out - self.feature_center) / self.feature_scale.replace(0, 1.0)
        elif self.feature_norm == "ts_zscore":
            assert isinstance(self.feature_center, pd.DataFrame)
            assert isinstance(self.feature_scale, pd.DataFrame)
            out = out.groupby(level="instrument", group_keys=False).apply(self._transform_one_instrument)
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if self.feature_clip is not None:
            out = out.clip(lower=-self.feature_clip, upper=self.feature_clip)
        return out.astype(np.float32)

    def _transform_one_instrument(self, frame: pd.DataFrame) -> pd.DataFrame:
        inst = frame.index.get_level_values("instrument")[0]
        assert isinstance(self.feature_center, pd.DataFrame)
        assert isinstance(self.feature_scale, pd.DataFrame)
        if inst not in self.feature_center.index:
            return frame * 0.0
        return (frame - self.feature_center.loc[inst]) / self.feature_scale.loc[inst].replace(0, 1.0)

    def transform_labels(self, label: pd.Series) -> pd.Series:
        out = label.copy()
        if self.label_center is not None and self.label_scale is not None:
            out = (out - self.label_center) / max(self.label_scale, 1e-6)
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if self.label_clip is not None:
            out = out.clip(lower=-self.label_clip, upper=self.label_clip)
        return out.astype(np.float32)


def get_feature_label_frame(dataset, segment: str) -> dict[str, pd.DataFrame | pd.Series]:
    df = dataset.prepare(segment, col_set=["feature", "label"], data_key="learn")
    if df is None or df.empty:
        raise ValueError(f"segment={segment} is empty")
    feature = df["feature"] if isinstance(df.columns, pd.MultiIndex) else df.iloc[:, :-1]
    label = df["label"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df.iloc[:, -1]
    common = feature.index.intersection(label.dropna().index)
    return {
        "feature": feature.loc[common].replace([np.inf, -np.inf], 0.0).fillna(0.0),
        "label": label.loc[common].replace([np.inf, -np.inf], 0.0).fillna(0.0),
    }


def get_feature_frame(dataset, segment: str) -> pd.DataFrame:
    df = dataset.prepare(segment, col_set=["feature"], data_key="infer")
    if df is None or df.empty:
        raise ValueError(f"segment={segment} is empty")
    return (df["feature"] if isinstance(df.columns, pd.MultiIndex) else df).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_predict_mlp(
    args: argparse.Namespace,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    test_x: pd.DataFrame,
) -> pd.Series:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], args.hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(args.hidden_dim, args.hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(args.hidden_dim // 2, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(train_x.to_numpy(np.float32)), torch.from_numpy(train_y.to_numpy(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_x_t = torch.from_numpy(valid_x.to_numpy(np.float32)).to(device)
    valid_y_t = torch.from_numpy(valid_y.to_numpy(np.float32)).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            count += len(yb)
        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x_t).squeeze(-1)
            valid_loss = float(loss_fn(valid_pred, valid_y_t).detach())
        print(f"epoch={epoch:03d} train_loss={total / max(count, 1):.6f} valid_loss={valid_loss:.6f}")

    model.eval()
    preds: list[np.ndarray] = []
    test_arr = test_x.to_numpy(np.float32)
    with torch.no_grad():
        for start in range(0, len(test_arr), args.batch_size):
            xb = torch.from_numpy(test_arr[start : start + args.batch_size]).to(device)
            preds.append(model(xb).squeeze(-1).cpu().numpy())
    return pd.Series(np.concatenate(preds), index=test_x.index, name="score")


def fit_predict_xgboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    test_x: pd.DataFrame,
) -> pd.Series:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run `pip install xgboost` on the server.") from exc

    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=8,
        early_stopping_rounds=30,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=50,
    )
    return pd.Series(model.predict(test_x), index=test_x.index, name="score")


if __name__ == "__main__":
    main()
