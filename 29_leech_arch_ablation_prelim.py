from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_WEIGHT_DECAYS = [0.22, 0.30]


def run(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short Leech architecture-ablation prelim. This is designed to "
            "probe one new lambda_geo setting against a small representative weight-"
            "decay slice, assuming the full lambda_geo=0.01 control already exists."
        )
    )
    parser.add_argument("--task", choices=["mul", "add"], default="mul")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--grok-thr", type=float, default=1.0)
    parser.add_argument("--grok-patience-logs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=DEFAULT_WEIGHT_DECAYS)
    parser.add_argument("--lambda-geo", type=float, default=0.0)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument("--output-root", type=str, default="leech_arch_ablation_prelim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "28_leech_arch_ablation.py",
        "--task",
        args.task,
        "--optimizer",
        args.optimizer,
        "--epochs",
        str(args.epochs),
        "--log-every",
        str(args.log_every),
        "--grok-thr",
        str(args.grok_thr),
        "--grok-patience-logs",
        str(args.grok_patience_logs),
        "--p",
        str(args.p),
        "--train-frac",
        str(args.train_frac),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--n-layers",
        str(args.n_layers),
        "--ff-mult",
        str(args.ff_mult),
        "--dropout",
        str(args.dropout),
        "--torch-device",
        args.torch_device,
        "--output-root",
        str(output_root),
        "--lambda-geos",
        str(args.lambda_geo),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--weight-decays",
        *[str(wd) for wd in args.weight_decays],
    ]
    run(cmd, cwd)


if __name__ == "__main__":
    main()
