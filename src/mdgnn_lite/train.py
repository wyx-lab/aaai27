from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

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
    parser.add_argument("--window", type=int, default=10)
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
    )
    print(
        "dataset: "
        f"dates={len(dataset.meta.dates)} instruments={len(dataset.meta.instruments)} "
        f"feature_dim={dataset.meta.feature_dim} samples={len(dataset)} "
        f"feature_nan_filled={dataset.meta.feature_nan_count} "
        f"label_nan_filled={dataset.meta.label_nan_count}"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        skipped = 0
        for batch_idx, batch in enumerate(loader):
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
                raise FloatingPointError("Gradient norm became NaN or Inf")
            optim.step()

            total_loss += float(loss.detach()) * int(mask.sum())
            total_count += int(mask.sum())
        print(
            f"epoch={epoch:03d} loss={total_loss / max(total_count, 1):.6f} "
            f"valid_count={total_count} skipped_batches={skipped}"
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


if __name__ == "__main__":
    main()
