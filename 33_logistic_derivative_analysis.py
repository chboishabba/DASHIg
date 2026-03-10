from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


TrajectoryKey = Tuple[int, float, int]


def load_summary(path: Path) -> Dict[TrajectoryKey, Dict[str, float]]:
    out: Dict[TrajectoryKey, Dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
            out[key] = {
                "t50": float(row["t50"]) if row["t50"] else math.nan,
                "t95": float(row["t95"]) if row["t95"] else math.nan,
            }
    return out


def load_trajectories(path: Path) -> Dict[TrajectoryKey, List[Dict[str, float]]]:
    out: Dict[TrajectoryKey, List[Dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
            out[key].append(
                {
                    "epoch": float(row["epoch"]),
                    "test_acc": float(row["test_acc"]),
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


def interpolate(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, xs, ys, left=ys[0], right=ys[-1])


def derivative_profile(
    epochs: np.ndarray,
    acc: np.ndarray,
    t50: float,
    smooth_window: int,
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = epochs / t50
    y = moving_average(acc, smooth_window)
    interp_y = interpolate(x, y, grid)
    deriv = np.gradient(interp_y, grid)
    return interp_y, deriv


def half_max_width(grid: np.ndarray, values: np.ndarray) -> float:
    peak = float(np.max(values))
    if peak <= 0:
        return math.nan
    half = peak / 2.0
    idx = np.where(values >= half)[0]
    if len(idx) < 2:
        return math.nan
    return float(grid[idx[-1]] - grid[idx[0]])


def derivative_skew(grid: np.ndarray, deriv: np.ndarray) -> float:
    positive = np.clip(deriv, 0.0, None)
    total = float(np.sum(positive))
    if total <= 0:
        return math.nan
    mean = float(np.sum(grid * positive) / total)
    centered = grid - mean
    var = float(np.sum((centered ** 2) * positive) / total)
    if var <= 0:
        return math.nan
    third = float(np.sum((centered ** 3) * positive) / total)
    return third / (var ** 1.5)


def gaussian_fit_mse(grid: np.ndarray, deriv: np.ndarray) -> Tuple[float, float, float]:
    positive = np.clip(deriv, 0.0, None)
    total = float(np.sum(positive))
    if total <= 0:
        return math.nan, math.nan, math.nan
    mu = float(np.sum(grid * positive) / total)
    var = float(np.sum(((grid - mu) ** 2) * positive) / total)
    sigma = math.sqrt(var) if var > 0 else math.nan
    amp = float(np.max(positive))
    if not math.isfinite(sigma) or sigma <= 0:
        return mu, sigma, math.nan
    fit = amp * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    mse = float(np.mean((positive - fit) ** 2))
    return mu, sigma, mse


def positive_area(grid: np.ndarray, deriv: np.ndarray, x_max: float | None = None) -> float:
    positive = np.clip(deriv, 0.0, None)
    if x_max is not None:
        mask = grid <= x_max
        if not np.any(mask):
            return math.nan
        return float(np.trapezoid(positive[mask], grid[mask]))
    return float(np.trapezoid(positive, grid))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, per_run: List[Dict[str, object]], summary: Dict[str, object], label: str) -> None:
    with path.open("w") as f:
        f.write("# Derivative Shape Analysis\n\n")
        f.write(f"- label: `{label}`\n\n")
        f.write("| weight_decay | seed | peak_x | peak_height | slope_proxy_k | half_max_width | pre_t50_area | skew | gaussian_mu | gaussian_sigma | gaussian_mse | corr_to_mean |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in per_run:
            f.write(
                f"| {row['weight_decay']} | {row['seed']} | {row['peak_x']} | {row['peak_height']} | "
                f"{row['slope_proxy_k']} | {row['half_max_width']} | {row['pre_t50_area']} | {row['skew']} | "
                f"{row['gaussian_mu']} | {row['gaussian_sigma']} | {row['gaussian_mse']} | {row['corr_to_mean']} |\n"
            )
        f.write("\n")
        f.write("## Summary\n\n")
        for key, value in summary.items():
            f.write(f"- {key}: {value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--grid-min", type=float, default=0.0)
    parser.add_argument("--grid-max", type=float, default=1.8)
    parser.add_argument("--grid-size", type=int, default=241)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--peak-margin", type=float, default=0.1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out-prefix", default="derivative_shape")
    args = parser.parse_args()

    summary_rows = load_summary(Path(args.summary))
    trajectories = load_trajectories(Path(args.trajectories))
    grid = np.linspace(args.grid_min, args.grid_max, args.grid_size)

    per_run_rows: List[Dict[str, object]] = []
    deriv_profiles: List[np.ndarray] = []
    keys_used: List[TrajectoryKey] = []

    for key in sorted(trajectories):
        t50 = summary_rows.get(key, {}).get("t50", math.nan)
        if not math.isfinite(t50) or t50 <= 0:
            continue
        rows = trajectories[key]
        epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
        acc = np.asarray([row["test_acc"] for row in rows], dtype=float)
        _, deriv = derivative_profile(epochs, acc, t50, args.smooth_window, grid)
        deriv_profiles.append(deriv)
        keys_used.append(key)

    if not deriv_profiles:
        raise SystemExit("no runs with finite t50 available")

    mean_profile = np.mean(np.stack(deriv_profiles, axis=0), axis=0)

    for key, deriv in zip(keys_used, deriv_profiles):
        p, weight_decay, seed = key
        valid_peak = np.where((grid >= args.peak_margin) & (grid <= args.grid_max - args.peak_margin))[0]
        if len(valid_peak) == 0:
            valid_peak = np.arange(len(grid))
        peak_idx = int(valid_peak[np.argmax(deriv[valid_peak])])
        peak_x = float(grid[peak_idx])
        peak_height = float(deriv[peak_idx])
        slope_proxy_k = 4.0 * peak_height
        width = half_max_width(grid, deriv)
        pre_t50_area = positive_area(grid, deriv, x_max=1.0)
        total_area = positive_area(grid, deriv)
        pre_t50_fraction = pre_t50_area / total_area if math.isfinite(total_area) and total_area > 0 else math.nan
        skew = derivative_skew(grid, deriv)
        g_mu, g_sigma, g_mse = gaussian_fit_mse(grid, deriv)
        corr = float(np.corrcoef(deriv, mean_profile)[0, 1]) if np.std(deriv) > 0 and np.std(mean_profile) > 0 else math.nan
        per_run_rows.append(
            {
                "p": p,
                "weight_decay": weight_decay,
                "seed": seed,
                "peak_x": peak_x,
                "peak_height": peak_height,
                "slope_proxy_k": slope_proxy_k,
                "half_max_width": width,
                "pre_t50_area": pre_t50_area,
                "pre_t50_fraction": pre_t50_fraction,
                "skew": skew,
                "gaussian_mu": g_mu,
                "gaussian_sigma": g_sigma,
                "gaussian_mse": g_mse,
                "corr_to_mean": corr,
            }
        )

    summary = {
        "label": args.label or Path(args.out_prefix).name,
        "mean_peak_x": float(np.mean([row["peak_x"] for row in per_run_rows])),
        "std_peak_x": float(np.std([row["peak_x"] for row in per_run_rows])),
        "mean_slope_proxy_k": float(np.mean([row["slope_proxy_k"] for row in per_run_rows])),
        "mean_half_max_width": float(np.mean([row["half_max_width"] for row in per_run_rows])),
        "mean_pre_t50_area": float(np.mean([row["pre_t50_area"] for row in per_run_rows])),
        "mean_pre_t50_fraction": float(np.mean([row["pre_t50_fraction"] for row in per_run_rows])),
        "mean_skew": float(np.mean([row["skew"] for row in per_run_rows])),
        "mean_gaussian_mse": float(np.mean([row["gaussian_mse"] for row in per_run_rows])),
        "mean_corr_to_mean": float(np.mean([row["corr_to_mean"] for row in per_run_rows])),
        "n_runs": len(per_run_rows),
    }

    summary_csv = Path(f"{args.out_prefix}_per_run.csv")
    summary_md = Path(f"{args.out_prefix}.md")
    summary_summary_csv = Path(f"{args.out_prefix}_summary.csv")

    write_csv(summary_csv, list(per_run_rows[0].keys()), per_run_rows)
    write_md(summary_md, per_run_rows, summary, summary["label"])
    write_csv(summary_summary_csv, list(summary.keys()), [summary])
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_summary_csv}")
    print(f"Saved: {summary_md}")



if __name__ == "__main__":
    main()
