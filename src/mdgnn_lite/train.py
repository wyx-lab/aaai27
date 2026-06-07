from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import Alpha158WindowDataset
from .graph import identity_relation, load_relation_tensor
from .model import MDGNNLite, tensor_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MDGNN-lite on exported Alpha158 parquet files.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--relations", default=None, help="Optional .npy/.npz/.csv relation graph.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-start", default=None, help="Target-date training start. Defaults to first sample.")
    parser.add_argument("--train-end", default=None, help="Target-date training end. Defaults to before validation.")
    parser.add_argument("--valid-start", default=None, help="Target-date validation start. Defaults to final 20%.")
    parser.add_argument("--valid-end", default=None, help="Target-date validation end. Defaults to last sample.")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation ratio when valid-start is absent.")
    parser.add_argument("--feature-norm", choices=["zscore", "robust", "none"], default="zscore")
    parser.add_argument("--feature-clip", type=float, default=10.0, help="Clip normalized feature values.")
    parser.add_argument("--label-clip", type=float, default=0.2, help="Clip label values before training.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/mdgnn_lite.pt")
    parser.add_argument("--debug", action="store_true", help="Print tensor diagnostics during training.")
    parser.add_argument("--debug-batches", type=int, default=2, help="Number of initial batches to debug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Alpha158WindowDataset(
        args.features,
        args.labels,
        start=args.start,
        end=args.end,
        window=args.window,
        label_clip=args.label_clip,
    )
    train_indices, valid_indices = split_indices_by_date(dataset, args)
    dataset.normalize_features(train_indices, mode=args.feature_norm, clip=args.feature_clip)
    print(
        "dataset: "
        f"dates={len(dataset.meta.dates)} instruments={len(dataset.meta.instruments)} "
        f"feature_dim={dataset.meta.feature_dim} samples={len(dataset)} "
        f"feature_norm={dataset.meta.feature_norm} "
        f"feature_nan_filled={dataset.meta.feature_nan_count} "
        f"label_nan_filled={dataset.meta.label_nan_count} "
        f"feature_clipped={dataset.meta.feature_clip_count} "
        f"label_clipped={dataset.meta.label_clip_count}"
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=args.batch_size, shuffle=False, drop_last=False)
    print_side_info(dataset, train_indices, valid_indices, args)

    if args.relations:
        relations = load_relation_tensor(args.relations, dataset.meta.instruments)
    else:
        relations = identity_relation(len(dataset.meta.instruments))
    relations = relations.to(args.device)
    if not torch.isfinite(relations).all():
        print(tensor_debug("relations", relations))
        raise ValueError("Relation tensor contains NaN or Inf")

    model = MDGNNLite(
        feature_dim=dataset.meta.feature_dim,
        num_relations=relations.shape[0],
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss(reduction="none")
    print("loss_fn: masked MSELoss(reduction='none').mean()")
    print("metrics: validation loss, daily cross-sectional IC, daily cross-sectional RankIC")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        skipped = 0
        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)
            mask = batch["mask"].to(args.device)
            valid_count = int(mask.sum().item())
            should_debug = args.debug and (batch_idx < args.debug_batches)
            if should_debug:
                print(f"batch={batch_idx} valid_count={valid_count}/{mask.numel()}")
                print(tensor_debug("batch.x", x))
                print(tensor_debug("batch.y", y))
                print(tensor_debug("batch.mask.float", mask.float()))
            if valid_count == 0:
                skipped += 1
                if should_debug:
                    print(f"batch={batch_idx} skipped because mask has no valid entries")
                continue

            pred = model(x, relations, debug=should_debug)
            loss = loss_fn(pred, y)
            if should_debug:
                print(tensor_debug("batch.pred", pred))
                print(tensor_debug("batch.loss_raw", loss))
            loss = loss[mask].mean()
            if not torch.isfinite(loss):
                print("non-finite loss detected")
                print(f"epoch={epoch} batch={batch_idx} valid_count={valid_count}/{mask.numel()}")
                print(tensor_debug("x", x))
                print(tensor_debug("y", y))
                print(tensor_debug("pred", pred))
                print(tensor_debug("loss_raw", loss_fn(pred, y)))
                print(tensor_debug("relations", relations))
                raise FloatingPointError("Loss became NaN or Inf")

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            if should_debug:
                print(f"batch={batch_idx} loss={float(loss.detach()):.6g} grad_norm={float(grad_norm):.6g}")
            if not torch.isfinite(grad_norm):
                print_nonfinite_grads(model)
                raise FloatingPointError("Gradient norm became NaN or Inf")
            optim.step()

            total_loss += float(loss.detach()) * int(mask.sum())
            total_count += int(mask.sum())
        print(
            f"epoch={epoch:03d} loss={total_loss / max(total_count, 1):.6f} "
            f"valid_count={total_count} skipped_batches={skipped}"
        )
        valid_stats = evaluate(model, valid_loader, relations, loss_fn, args.device)
        print(
            f"epoch={epoch:03d} valid_loss={valid_stats['loss']:.6f} "
            f"valid_ic={valid_stats['ic']:.6f} valid_rankic={valid_stats['rankic']:.6f} "
            f"valid_count={valid_stats['count']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "instruments": dataset.meta.instruments,
            "feature_dim": dataset.meta.feature_dim,
            "num_relations": int(relations.shape[0]),
            "args": vars(args),
        },
        out,
    )
    print(f"saved: {out.resolve()}")


def split_indices_by_date(dataset: Alpha158WindowDataset, args: argparse.Namespace) -> tuple[list[int], list[int]]:
    target_dates = dataset.sample_target_dates()
    all_indices = list(range(len(target_dates)))
    valid_start = args.valid_start
    valid_end = args.valid_end
    train_start = args.train_start
    train_end = args.train_end

    if valid_start is None:
        valid_size = max(1, int(len(all_indices) * args.valid_ratio))
        valid_start_idx = max(0, len(all_indices) - valid_size)
        valid_start = str(target_dates[valid_start_idx].date())

    train_indices: list[int] = []
    valid_indices: list[int] = []
    for idx, date in enumerate(target_dates):
        if train_start is not None and date < torch_timestamp(train_start):
            continue
        if valid_start is not None and date >= torch_timestamp(valid_start):
            if valid_end is None or date <= torch_timestamp(valid_end):
                valid_indices.append(idx)
            continue
        if train_end is None or date <= torch_timestamp(train_end):
            train_indices.append(idx)

    if not train_indices:
        raise ValueError("No training samples after date split")
    if not valid_indices:
        raise ValueError("No validation samples after date split")
    return train_indices, valid_indices


def torch_timestamp(value: str):
    import pandas as pd

    return pd.Timestamp(value)


def print_side_info(
    dataset: Alpha158WindowDataset,
    train_indices: list[int],
    valid_indices: list[int],
    args: argparse.Namespace,
) -> None:
    target_dates = dataset.sample_target_dates()
    first_item = dataset[train_indices[0]]
    valid_item = dataset[valid_indices[0]]
    print(
        "split: "
        f"train_samples={len(train_indices)} valid_samples={len(valid_indices)} "
        f"train_range={target_dates[train_indices[0]].date()}..{target_dates[train_indices[-1]].date()} "
        f"valid_range={target_dates[valid_indices[0]].date()}..{target_dates[valid_indices[-1]].date()}"
    )
    print(
        "batch_shapes: "
        f"x_per_sample={tuple(first_item['x'].shape)} y_per_sample={tuple(first_item['y'].shape)} "
        f"mask_per_sample={tuple(first_item['mask'].shape)} batch_size={args.batch_size}"
    )
    print(
        "valid_shapes: "
        f"x_per_sample={tuple(valid_item['x'].shape)} y_per_sample={tuple(valid_item['y'].shape)} "
        f"mask_per_sample={tuple(valid_item['mask'].shape)}"
    )


@torch.no_grad()
def evaluate(
    model: MDGNNLite,
    loader: DataLoader,
    relations: torch.Tensor,
    loss_fn: nn.Module,
    device: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    daily_ic: list[float] = []
    daily_rankic: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x, relations)
        loss = loss_fn(pred, y)
        total_loss += float(loss[mask].sum().detach())
        total_count += int(mask.sum().item())
        daily_ic.extend(batch_corr(pred, y, mask, rank=False))
        daily_rankic.extend(batch_corr(pred, y, mask, rank=True))
    return {
        "loss": total_loss / max(total_count, 1),
        "ic": safe_mean(daily_ic),
        "rankic": safe_mean(daily_rankic),
        "count": float(total_count),
    }


def batch_corr(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, rank: bool) -> list[float]:
    values: list[float] = []
    for p_row, y_row, m_row in zip(pred.detach().cpu(), y.detach().cpu(), mask.detach().cpu()):
        p = p_row[m_row].float()
        t = y_row[m_row].float()
        if p.numel() < 2:
            continue
        if rank:
            p = rank_tensor(p)
            t = rank_tensor(t)
        p = p - p.mean()
        t = t - t.mean()
        denom = torch.sqrt((p.square().sum() * t.square().sum()).clamp_min(1e-12))
        corr = float((p * t).sum() / denom)
        if torch.isfinite(torch.tensor(corr)):
            values.append(corr)
    return values


def rank_tensor(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=x.dtype)
    return ranks


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def print_nonfinite_grads(model: nn.Module) -> None:
    print("non-finite gradients detected by parameter:")
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if torch.isfinite(grad).all():
            continue
        finite = torch.isfinite(grad)
        finite_count = int(finite.sum().item())
        total = grad.numel()
        print(tensor_debug(f"grad.{name}", grad))
        print(f"grad.{name}: finite={finite_count}/{total}")


if __name__ == "__main__":
    main()
