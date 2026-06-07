from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Alpha158WindowDataset
from .graph import identity_relation, load_relation_tensor
from .model import MDGNNLite


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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.relations:
        relations = load_relation_tensor(args.relations, dataset.meta.instruments)
    else:
        relations = identity_relation(len(dataset.meta.instruments))
    relations = relations.to(args.device)

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
        for batch in loader:
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)
            mask = batch["mask"].to(args.device)
            pred = model(x, relations)
            loss = loss_fn(pred, y)
            loss = loss[mask].mean()

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()

            total_loss += float(loss.detach()) * int(mask.sum())
            total_count += int(mask.sum())
        print(f"epoch={epoch:03d} loss={total_loss / max(total_count, 1):.6f}")

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
