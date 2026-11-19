"""6-fold leave-one-guitarist-out TabCNN experiment.

PyTorch port of the experiment harness at the bottom of the original
``TabCNN.py`` (which used Keras ``fit_generator``). For each of the 6 folds it
trains on 5 guitarists, evaluates on the held-out one, and writes weights,
predictions, and an aggregated ``results.csv``.

Usage:
    python src/tabcnn/train.py --data_path ../data/spec_repr/ --epochs 8
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TabFrameDataset, load_ids, partition_ids
from metrics import (
    pitch_f_measure,
    pitch_precision,
    pitch_recall,
    tab_disamb,
    tab_f_measure,
    tab_precision,
    tab_recall,
)
from tabcnn import SPEC_REPR_BINS, TabCNN, avg_acc, predict_proba, tab_loss

METRIC_FNS = {
    "pp": pitch_precision, "pr": pitch_recall, "pf": pitch_f_measure,
    "tp": tab_precision, "tr": tab_recall, "tf": tab_f_measure,
    "tdr": tab_disamb,
}


def train_one_fold(model, loader, optimizer, epochs, device, log_every=0):
    model.train()
    n_steps = len(loader)
    for epoch in range(epochs):
        running_loss = running_acc = n_batches = 0
        t_epoch = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = tab_loss(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += avg_acc(logits, y).item()
            n_batches += 1
            # Optional within-epoch progress so long epochs aren't silent.
            if log_every and n_batches % log_every == 0:
                rate = n_batches / (time.time() - t_epoch)
                print(f"    epoch {epoch + 1}/{epochs}  step {n_batches}/{n_steps}  "
                      f"loss {running_loss / n_batches:.4f}  "
                      f"acc {running_acc / n_batches:.4f}  "
                      f"{rate:.1f} it/s", flush=True)
        print(f"  epoch {epoch + 1}/{epochs}  "
              f"loss {running_loss / n_batches:.4f}  "
              f"acc {running_acc / n_batches:.4f}  "
              f"({time.time() - t_epoch:.0f} s)", flush=True)


@torch.no_grad()
def predict_test(model, dataset, batch_size, device):
    """Run the model over the whole test set, returning (y_pred, y_gt)
    as NumPy arrays shaped (N, 6, 21)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, gts = [], []
    for x, y in loader:
        preds.append(predict_proba(model, x.to(device)).cpu().numpy())
        gts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", default="../data/spec_repr/")
    parser.add_argument("--id_file", default="id.csv")
    parser.add_argument("--spec_repr", default="c", choices=sorted(SPEC_REPR_BINS))
    parser.add_argument("--con_win_size", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=0,
                        help="print progress every N steps within an epoch (0 = only per-epoch)")
    parser.add_argument("--save_path", default="saved/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = Path(args.save_path) / f"{args.spec_repr}_{stamp}"
    save_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    list_IDs = load_ids(args.data_path, args.id_file)
    metrics = {key: [] for key in METRIC_FNS}

    for fold in range(6):
        print(f"\nfold {fold}")
        split_folder = save_folder / str(fold)
        split_folder.mkdir(exist_ok=True)

        partition = partition_ids(list_IDs, fold)
        common = dict(data_path=args.data_path, spec_repr=args.spec_repr,
                      con_win_size=args.con_win_size)
        train_set = TabFrameDataset(partition["training"], **common)
        test_set = TabFrameDataset(partition["test"], **common)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)

        model = TabCNN().to(device)
        # Materialize the LazyLinear layer before handing params to the optimizer.
        model(torch.zeros(1, 1, SPEC_REPR_BINS[args.spec_repr],
                          args.con_win_size, device=device))
        # Matches Keras' Adadelta defaults (rho=0.95, eps=1e-7) rather than
        # PyTorch's (rho=0.9, eps=1e-6).
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-7)

        print("training...")
        train_one_fold(model, train_loader, optimizer, args.epochs, device, args.log_every)
        torch.save(model.state_dict(), split_folder / "weights.pt")

        print("testing...")
        y_pred, y_gt = predict_test(model, test_set, args.batch_size, device)
        np.savez(split_folder / "predictions.npz", y_pred=y_pred, y_gt=y_gt)

        print("evaluation...")
        for key, fn in METRIC_FNS.items():
            metrics[key].append(fn(y_pred, y_gt))

    # Aggregate per-fold metrics with mean and std, like the original results.csv.
    output = {"data": [f"g{i}" for i in range(6)] + ["mean", "std dev"]}
    for key, vals in metrics.items():
        output[key] = vals + [float(np.mean(vals)), float(np.std(vals))]
    pd.DataFrame(output).to_csv(save_folder / "results.csv", index=False)
    print(f"\nsaved results to {save_folder / 'results.csv'}")


if __name__ == "__main__":
    main()
