from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


SeriesKey = Tuple[float, int]


def load_trajectories(path: Path, value_field: str) -> Dict[SeriesKey, List[Dict[str, float]]]:
    out: Dict[SeriesKey, List[Dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["weight_decay"]), int(row["seed"]))
            out[key].append(
                {
                    "epoch": float(row["epoch"]),
                    "value": float(row[value_field]),
                }
            )
    for rows in out.values():
        rows.sort(key=lambda r: r["epoch"])
    return out


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def dominant_frequency(
    epochs: np.ndarray,
    values: np.ndarray,
    smooth_window: int,
    min_period_epochs: float,
) -> Dict[str, float]:
    if len(epochs) < 8:
        return {
            "spacing": math.nan,
            "dominant_period_epochs": math.nan,
            "dominant_frequency_per_epoch": math.nan,
            "peak_power_ratio": math.nan,
            "n_points": float(len(epochs)),
        }

    diffs = np.diff(epochs)
    step = float(np.median(diffs)) if len(diffs) else 1.0
    if step <= 0:
        step = 1.0

    smoothed = moving_average(values, smooth_window)
    residual = values - smoothed
    residual = residual - np.mean(residual)
    spectrum = np.fft.rfft(residual)
    freqs = np.fft.rfftfreq(len(residual), d=step)
    power = np.abs(spectrum) ** 2

    if len(freqs) <= 1:
        return {
            "spacing": step,
            "dominant_period_epochs": math.nan,
            "dominant_frequency_per_epoch": math.nan,
            "peak_power_ratio": math.nan,
            "n_points": float(len(epochs)),
        }

    power[0] = 0.0
    min_freq = 1.0 / max(min_period_epochs, step)
    valid = np.where(freqs >= min_freq)[0]
    if len(valid) == 0:
        valid = np.arange(1, len(freqs))
    valid_power = power[valid]
    if not np.any(valid_power > 0):
        return {
            "spacing": step,
            "dominant_period_epochs": math.nan,
            "dominant_frequency_per_epoch": math.nan,
            "peak_power_ratio": 0.0,
            "n_points": float(len(epochs)),
        }

    best_idx = valid[int(np.argmax(valid_power))]
    dominant_freq = float(freqs[best_idx])
    dominant_period = float(1.0 / dominant_freq) if dominant_freq > 0 else math.nan
    peak_ratio = float(power[best_idx] / np.sum(valid_power)) if np.sum(valid_power) > 0 else math.nan
    return {
        "spacing": step,
        "dominant_period_epochs": dominant_period,
        "dominant_frequency_per_epoch": dominant_freq,
        "peak_power_ratio": peak_ratio,
        "n_points": float(len(epochs)),
    }


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, object]], value_field: str, smooth_window: int, min_period_epochs: float) -> None:
    with path.open("w") as f:
        f.write("# FFT Spike Analysis\n\n")
        f.write(f"- value field: `{value_field}`\n")
        f.write(f"- smoothing window: `{smooth_window}` samples\n")
        f.write(f"- minimum period searched: `{min_period_epochs}` epochs\n\n")
        f.write("| weight_decay | seed | spacing | dominant period (epochs) | peak power ratio |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['weight_decay']} | {row['seed']} | {row['sample_spacing_epochs']} | "
                f"{row['dominant_period_epochs']} | {row['peak_power_ratio']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--value-field", default="test_loss", choices=["test_loss", "train_loss", "test_acc", "train_acc"])
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--min-period-epochs", type=float, default=200.0)
    parser.add_argument("--out-prefix", default="fft_spike_analysis")
    args = parser.parse_args()

    series = load_trajectories(Path(args.trajectories), args.value_field)
    rows: List[Dict[str, object]] = []
    for (weight_decay, seed), pts in sorted(series.items()):
        epochs = np.asarray([pt["epoch"] for pt in pts], dtype=float)
        values = np.asarray([pt["value"] for pt in pts], dtype=float)
        stats = dominant_frequency(epochs, values, args.smooth_window, args.min_period_epochs)
        rows.append(
            {
                "weight_decay": weight_decay,
                "seed": seed,
                "value_field": args.value_field,
                "sample_spacing_epochs": stats["spacing"],
                "dominant_period_epochs": stats["dominant_period_epochs"],
                "dominant_frequency_per_epoch": stats["dominant_frequency_per_epoch"],
                "peak_power_ratio": stats["peak_power_ratio"],
                "n_points": int(stats["n_points"]),
            }
        )

    csv_path = Path(f"{args.out_prefix}.csv")
    md_path = Path(f"{args.out_prefix}.md")
    write_csv(csv_path, list(rows[0].keys()) if rows else ["weight_decay"], rows)
    write_md(md_path, rows, args.value_field, args.smooth_window, args.min_period_epochs)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
