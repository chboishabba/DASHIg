from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from phase2_validation.leech_modular_benchmark import LeechBenchmarkConfig, run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mul", "add"], default="mul")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.20, 0.22, 0.24, 0.25, 0.30, 0.35, 0.40])
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--lambda-geo", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--out-prefix", type=str, default="leech_grok_critical_scan")
    parser.add_argument("--torch-device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.torch_device if args.torch_device != "cuda" or torch.cuda.is_available() else "cpu"

    rows = []
    trajectories = []

    for weight_decay in args.weight_decays:
        for seed in args.seeds:
            cfg = LeechBenchmarkConfig(
                p=args.p,
                task=args.task,
                optimizer=args.optimizer,
                seed=seed,
                train_frac=args.train_frac,
                epochs=args.epochs,
                log_every=args.log_every,
                lr=args.lr,
                weight_decay=weight_decay,
                momentum=args.momentum,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                ff_mult=args.ff_mult,
                lambda_geo=args.lambda_geo,
                dropout=args.dropout,
                device=device,
            )
            summary, trajectory = run_one(cfg)
            rows.append(summary)
            trajectories.extend(trajectory)
            print(
                f"task={args.task} opt={args.optimizer} wd={weight_decay:<4} seed={seed:<2} "
                f"t50={summary['t50']} t95={summary['t95']} "
                f"train={summary['final_train_acc']:.3f} test={summary['final_test_acc']:.3f}"
            )

    summary_path = Path(f"{args.out_prefix}.csv")
    traj_path = Path(f"{args.out_prefix}_trajectories.csv")
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectories[0].keys()))
        writer.writeheader()
        writer.writerows(trajectories)

    print(f"Saved: {summary_path}")
    print(f"Saved: {traj_path}")


if __name__ == "__main__":
    main()
