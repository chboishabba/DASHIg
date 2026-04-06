from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def derivative(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values)


def load_series(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)
    if "step" not in df.columns or "train_loss" not in df.columns:
        raise ValueError(f"{csv_path} must contain step and train_loss columns")
    val = None
    if "val_loss" in df.columns:
        val_series = pd.to_numeric(df["val_loss"], errors="coerce")
        if val_series.notna().any():
            val = val_series.to_numpy(dtype=float)
    return (
        df["step"].to_numpy(dtype=float),
        df["train_loss"].to_numpy(dtype=float),
        val,
    )


def plot_one(csv_path: Path, label: str) -> None:
    steps, train_loss, val_loss = load_series(csv_path)
    plt.plot(steps, train_loss, label=f"{label} train")
    plt.plot(steps, derivative(train_loss), linestyle="--", label=f"{label} d(train)")
    if val_loss is not None:
        plt.plot(steps, val_loss, alpha=0.7, label=f"{label} val")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw LILA/Leech training dynamics from adapted CSV logs.")
    parser.add_argument("--primary", required=True, help="Primary CSV log")
    parser.add_argument("--primary-label", default="lila", help="Primary plot label")
    parser.add_argument("--compare", default=None, help="Optional second CSV log")
    parser.add_argument("--compare-label", default="compare", help="Second plot label")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    plt.figure(figsize=(11, 6))
    plot_one(Path(args.primary), args.primary_label)
    if args.compare:
        plot_one(Path(args.compare), args.compare_label)
    plt.xlabel("step")
    plt.ylabel("loss / derivative")
    plt.title("LILA external training dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
