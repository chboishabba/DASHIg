#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares


TrajectoryKey = Tuple[int, float, int]


def load_summary(paths: List[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    seen = set()
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "p": int(row["p"]),
                        "seed": int(row["seed"]),
                        "weight_decay": float(row["weight_decay"]),
                        "epochs": float(row["epochs"]),
                        "train_frac": float(row["train_frac"]),
                        "lr": float(row["lr"]),
                        "t_fit": float(row["t_fit"]) if row["t_fit"] else math.nan,
                        "t50": float(row["t50"]) if row["t50"] else math.nan,
                        "t95": float(row["t95"]) if row["t95"] else math.nan,
                        "final_train_loss": float(row["final_train_loss"]),
                        "final_test_loss": float(row["final_test_loss"]),
                        "final_train_acc": float(row["final_train_acc"]),
                        "final_test_acc": float(row["final_test_acc"]),
                    }
                )
    return rows


def load_trajectories(paths: List[Path]) -> Dict[TrajectoryKey, List[Dict[str, float]]]:
    out: Dict[TrajectoryKey, List[Dict[str, float]]] = defaultdict(list)
    seen_rows = set()
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
                row_key = key + (float(row["epoch"]),)
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                out[key].append(
                    {
                        "epoch": float(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "test_loss": float(row["test_loss"]),
                        "train_acc": float(row["train_acc"]),
                        "test_acc": float(row["test_acc"]),
                    }
                )
    for rows in out.values():
        rows.sort(key=lambda r: r["epoch"])
    return out


def first_ge(rows: Iterable[Dict[str, float]], field: str, threshold: float) -> float:
    for row in rows:
        if row[field] >= threshold:
            return row["epoch"]
    return math.nan


def epoch_or_blank(value: float) -> int | str:
    return int(value) if math.isfinite(value) else ""


def parse_optional_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_linear(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    if not xs or not ys:
        return math.nan, math.nan, math.nan
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx == 0.0:
        return math.nan, math.nan, math.nan
    slope = float(np.sum((x - xbar) * (y - ybar)) / sxx)
    intercept = ybar - slope * xbar
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def interpolate(xs: List[float], ys: List[float], grid: List[float]) -> List[float]:
    out: List[float] = []
    for g in grid:
        if g <= xs[0]:
            out.append(ys[0])
            continue
        if g >= xs[-1]:
            out.append(ys[-1])
            continue
        j = 1
        while xs[j] < g:
            j += 1
        x0, x1 = xs[j - 1], xs[j]
        y0, y1 = ys[j - 1], ys[j]
        t = (g - x0) / (x1 - x0)
        out.append(y0 + t * (y1 - y0))
    return out


def normalized_alignment_mse(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    milestone_field: str,
) -> float:
    grid = [i / 20.0 for i in range(21)]
    curves: List[List[float]] = []
    for row in milestones:
        T = parse_optional_float(row[milestone_field])
        if not math.isfinite(T) or T <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        xs = [pt["epoch"] / T for pt in traj]
        ys = [pt["test_acc"] for pt in traj]
        curves.append(interpolate(xs, ys, grid))
    if not curves:
        return math.nan
    mean = [sum(curve[i] for curve in curves) / len(curves) for i in range(len(grid))]
    return sum((curve[i] - mean[i]) ** 2 for curve in curves for i in range(len(grid))) / (len(curves) * len(grid))


def plot_raw(trajectories: Dict[TrajectoryKey, List[Dict[str, float]]], y_field: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(trajectories):
        _, wd, seed = key
        rows = trajectories[key]
        ax.plot(
            [r["epoch"] for r in rows],
            [r[y_field] for r in rows],
            label=f"wd={wd:.2f}, seed={seed}",
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_field.replace("_", " "))
    ax.set_title(f"Raw {y_field.replace('_', ' ')} trajectories")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_normalized(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    milestone_field: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        T = parse_optional_float(row[milestone_field])
        if not math.isfinite(T) or T <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        ax.plot(
            [pt["epoch"] / T for pt in traj],
            [pt["test_acc"] for pt in traj],
            label=f"wd={float(row['weight_decay']):.2f}",
            linewidth=2,
        )
    ax.set_xlabel(f"Epoch / {milestone_field}")
    ax.set_ylabel("test acc")
    ax.set_title(f"Normalized test acc by {milestone_field}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def gompertz(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.exp(-a * np.exp(-b * x))


def logistic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def curvature_onset_epoch(rows: List[Dict[str, float]]) -> float:
    if len(rows) < 3:
        return math.nan
    best_epoch = math.nan
    best_curvature = -math.inf
    for idx in range(1, len(rows) - 1):
        y_prev = rows[idx - 1]["test_acc"]
        y_curr = rows[idx]["test_acc"]
        y_next = rows[idx + 1]["test_acc"]
        curvature = y_next - 2.0 * y_curr + y_prev
        if curvature > best_curvature:
            best_curvature = curvature
            best_epoch = rows[idx]["epoch"]
    return best_epoch


def normalized_points(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    milestone_field: str,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for row in milestones:
        T = parse_optional_float(row[milestone_field])
        if not math.isfinite(T) or T <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        xs.extend(pt["epoch"] / T for pt in traj)
        ys.extend(pt["test_acc"] for pt in traj)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_shape_laws(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    out_csv: Path,
    out_plot: Path,
) -> None:
    x, y = normalized_points(trajectories, milestones, "t50")
    if len(x) == 0:
        write_csv(out_csv, ["model", "param_1", "param_2", "mse"], [])
        return
    gom_popt, _ = curve_fit(gompertz, x, y, p0=(5.0, 5.0), bounds=((1e-6, 1e-6), (100.0, 100.0)), maxfev=20000)
    log_popt, _ = curve_fit(logistic, x, y, p0=(10.0, 1.0), bounds=((1e-6, -10.0), (100.0, 10.0)), maxfev=20000)
    mse_g = float(np.mean((y - gompertz(x, *gom_popt)) ** 2))
    mse_l = float(np.mean((y - logistic(x, *log_popt)) ** 2))
    write_csv(
        out_csv,
        ["model", "param_1", "param_2", "mse"],
        [
            {"model": "gompertz_norm_t50", "param_1": float(gom_popt[0]), "param_2": float(gom_popt[1]), "mse": mse_g},
            {"model": "logistic_norm_t50", "param_1": float(log_popt[0]), "param_2": float(log_popt[1]), "mse": mse_l},
        ],
    )
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        T = parse_optional_float(row["t50"])
        if not math.isfinite(T) or T <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        ax.plot([pt["epoch"] / T for pt in traj], [pt["test_acc"] for pt in traj], linewidth=1.6, alpha=0.75, label=f"wd={float(row['weight_decay']):.2f}")
    ax.plot(grid, gompertz(grid, *gom_popt), color="black", linewidth=2.5, label="gompertz fit")
    ax.plot(grid, logistic(grid, *log_popt), color="gray", linestyle="--", linewidth=2.0, label="logistic fit")
    ax.set_xlabel("Epoch / t50")
    ax.set_ylabel("test acc")
    ax.set_title("Normalized test acc with shared shape-law fits")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def fit_shifted_rising_logistic(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    out_csv: Path,
    out_plot: Path,
    shift_field: str,
    model_name: str,
    xlabel: str,
    title: str,
) -> None:
    xs: List[float] = []
    ys: List[float] = []
    per_curve: List[Tuple[float, List[float], List[float]]] = []
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        shift_epoch = parse_optional_float(row[shift_field])
        t50 = parse_optional_float(row["t50"])
        if not math.isfinite(shift_epoch) or not math.isfinite(t50) or t50 <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        curve_x: List[float] = []
        curve_y: List[float] = []
        for pt in traj:
            if pt["epoch"] < shift_epoch:
                continue
            x = (pt["epoch"] - shift_epoch) / t50
            curve_x.append(x)
            curve_y.append(pt["test_acc"])
            xs.append(x)
            ys.append(pt["test_acc"])
        if curve_x:
            per_curve.append((float(row["weight_decay"]), curve_x, curve_y))
    if not xs:
        write_csv(out_csv, ["model", "param_1", "param_2", "mse"], [])
        return
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    log_popt, _ = curve_fit(logistic, x_arr, y_arr, p0=(12.0, 0.4), bounds=((1e-6, -10.0), (100.0, 10.0)), maxfev=20000)
    mse = float(np.mean((y_arr - logistic(x_arr, *log_popt)) ** 2))
    write_csv(
        out_csv,
        ["model", "param_1", "param_2", "mse"],
        [{"model": model_name, "param_1": float(log_popt[0]), "param_2": float(log_popt[1]), "mse": mse}],
    )
    grid = np.linspace(0.0, max(max(curve_x) for _, curve_x, _ in per_curve), 400)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wd, curve_x, curve_y in per_curve:
        ax.plot(curve_x, curve_y, linewidth=1.6, alpha=0.75, label=f"wd={wd:.2f}")
    ax.plot(grid, logistic(grid, *log_popt), color="black", linewidth=2.5, label="rise logistic fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test acc")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def fit_rising_phase_logistic_fitted_t0(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    out_csv: Path,
    out_plot: Path,
) -> None:
    curve_data: List[Tuple[float, List[float], List[float]]] = []
    init_offsets: List[float] = []
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        t10 = parse_optional_float(row["t10"])
        t50 = parse_optional_float(row["t50"])
        if not math.isfinite(t10) or not math.isfinite(t50) or t50 <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        xs = [pt["epoch"] / t50 for pt in traj]
        ys = [pt["test_acc"] for pt in traj]
        curve_data.append((float(row["weight_decay"]), xs, ys))
        init_offsets.append(t10 / t50)
    if not curve_data:
        write_csv(out_csv, ["model", "param_1", "param_2", "mse"], [])
        return

    def residuals(params: np.ndarray) -> np.ndarray:
        k = params[0]
        x0 = params[1]
        shifts = params[2:]
        out: List[float] = []
        for idx, (_, xs, ys) in enumerate(curve_data):
            shift = shifts[idx]
            for x, y in zip(xs, ys):
                out.append(logistic(np.asarray([x - shift]), k, x0)[0] - y)
        return np.asarray(out, dtype=float)

    x0_init = np.asarray([12.0, 0.2] + init_offsets, dtype=float)
    lower = np.asarray([1e-6, -10.0] + [0.0 for _ in init_offsets], dtype=float)
    upper = np.asarray([100.0, 10.0] + [10.0 for _ in init_offsets], dtype=float)
    result = least_squares(residuals, x0_init, bounds=(lower, upper), max_nfev=40000)
    params = result.x
    k = float(params[0])
    x0 = float(params[1])
    shifts = [float(v) for v in params[2:]]
    mse = float(np.mean(residuals(params) ** 2))
    rows: List[Dict[str, object]] = [{"model": "rise_logistic_norm_t50_fitted_t0", "param_1": k, "param_2": x0, "mse": mse}]
    for (wd, _, _), shift in zip(curve_data, shifts):
        rows.append({"model": f"shift_wd_{wd:.2f}", "param_1": shift, "param_2": "", "mse": ""})
    write_csv(out_csv, ["model", "param_1", "param_2", "mse"], rows)
    max_x = 0.0
    grid_min = math.inf
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (wd, xs, ys), shift in zip(curve_data, shifts):
        curve_x = [x - shift for x in xs]
        max_x = max(max_x, max(curve_x))
        grid_min = min(grid_min, min(curve_x))
        ax.plot(curve_x, ys, linewidth=1.6, alpha=0.75, label=f"wd={wd:.2f}")
    grid = np.linspace(grid_min, max_x if max_x > grid_min else grid_min + 1.0, 400)
    ax.plot(grid, logistic(grid, k, x0), color="black", linewidth=2.5, label="fitted-t0 logistic fit")
    ax.set_xlabel("(Epoch / t50) - fitted onset shift")
    ax.set_ylabel("test acc")
    ax.set_title("Rising-phase logistic fit with fitted onset shift")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def fit_rising_phase_logistic_fixed_ct50(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    out_csv: Path,
    out_plot: Path,
) -> None:
    curve_data: List[Tuple[float, List[float], List[float]]] = []
    init_cs: List[float] = []
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        t10 = parse_optional_float(row["t10"])
        t50 = parse_optional_float(row["t50"])
        if not math.isfinite(t10) or not math.isfinite(t50) or t50 <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        xs = [pt["epoch"] / t50 for pt in traj]
        ys = [pt["test_acc"] for pt in traj]
        curve_data.append((float(row["weight_decay"]), xs, ys))
        init_cs.append(t10 / t50)
    if not curve_data:
        write_csv(out_csv, ["model", "param_1", "param_2", "mse"], [])
        return

    def residuals(params: np.ndarray) -> np.ndarray:
        k = params[0]
        x0 = params[1]
        c = params[2]
        out: List[float] = []
        for _, xs, ys in curve_data:
            for x, y in zip(xs, ys):
                out.append(logistic(np.asarray([x - c]), k, x0)[0] - y)
        return np.asarray(out, dtype=float)

    p0 = np.asarray([12.0, 0.2, float(np.mean(init_cs))], dtype=float)
    lower = np.asarray([1e-6, -10.0, 0.0], dtype=float)
    upper = np.asarray([100.0, 10.0, 10.0], dtype=float)
    result = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=40000)
    k = float(result.x[0])
    x0 = float(result.x[1])
    c = float(result.x[2])
    mse = float(np.mean(residuals(result.x) ** 2))
    write_csv(
        out_csv,
        ["model", "param_1", "param_2", "mse"],
        [
            {"model": "rise_logistic_norm_t50_fixed_ct50", "param_1": k, "param_2": x0, "mse": mse},
            {"model": "shared_c", "param_1": c, "param_2": "", "mse": ""},
        ],
    )
    max_x = 0.0
    min_x = math.inf
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wd, xs, ys in curve_data:
        curve_x = [x - c for x in xs]
        max_x = max(max_x, max(curve_x))
        min_x = min(min_x, min(curve_x))
        ax.plot(curve_x, ys, linewidth=1.6, alpha=0.75, label=f"wd={wd:.2f}")
    grid = np.linspace(min_x, max_x if max_x > min_x else min_x + 1.0, 400)
    ax.plot(grid, logistic(grid, k, x0), color="black", linewidth=2.5, label="fixed c*t50 logistic fit")
    ax.set_xlabel("(Epoch / t50) - c")
    ax.set_ylabel("test acc")
    ax.set_title("Rising-phase logistic fit with shared normalized onset")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def fit_rising_phase_loss_logistic(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    out_csv: Path,
    out_plot: Path,
) -> None:
    xs: List[float] = []
    ys: List[float] = []
    per_curve: List[Tuple[float, List[float], List[float]]] = []
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        t10 = parse_optional_float(row["t10"])
        t50 = parse_optional_float(row["t50"])
        if not math.isfinite(t10) or not math.isfinite(t50) or t50 <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        post_rows = [pt for pt in trajectories[key] if pt["epoch"] >= t10]
        if len(post_rows) < 2:
            continue
        loss_start = float(post_rows[0]["test_loss"])
        loss_end = float(post_rows[-1]["test_loss"])
        denom = loss_start - loss_end
        if abs(denom) < 1e-12:
            continue
        curve_x: List[float] = []
        curve_y: List[float] = []
        for pt in post_rows:
            x = (pt["epoch"] - t10) / t50
            y = (loss_start - float(pt["test_loss"])) / denom
            y = min(1.0, max(0.0, y))
            curve_x.append(x)
            curve_y.append(y)
            xs.append(x)
            ys.append(y)
        per_curve.append((float(row["weight_decay"]), curve_x, curve_y))
    if not xs:
        write_csv(out_csv, ["model", "param_1", "param_2", "mse"], [])
        return
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    log_popt, _ = curve_fit(logistic, x_arr, y_arr, p0=(12.0, 0.4), bounds=((1e-6, -10.0), (100.0, 10.0)), maxfev=20000)
    mse = float(np.mean((y_arr - logistic(x_arr, *log_popt)) ** 2))
    write_csv(
        out_csv,
        ["model", "param_1", "param_2", "mse"],
        [{"model": "rise_loss_logistic_norm_t50_shift_t10", "param_1": float(log_popt[0]), "param_2": float(log_popt[1]), "mse": mse}],
    )
    grid = np.linspace(0.0, max(max(curve_x) for _, curve_x, _ in per_curve), 400)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wd, curve_x, curve_y in per_curve:
        ax.plot(curve_x, curve_y, linewidth=1.6, alpha=0.75, label=f"wd={wd:.2f}")
    ax.plot(grid, logistic(grid, *log_popt), color="black", linewidth=2.5, label="rise loss logistic fit")
    ax.set_xlabel("(Epoch - t10) / t50")
    ax.set_ylabel("normalized loss progress")
    ax.set_title("Rising-phase logistic fit on normalized test-loss progress")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", nargs="+", default=["leech_grok_critical_scan.csv"])
    parser.add_argument("--trajectories", nargs="+", default=["leech_grok_critical_scan_trajectories.csv"])
    parser.add_argument("--outdir", default="leech_grok_analysis")
    args = parser.parse_args()

    summary_rows = load_summary([Path(p) for p in args.summary])
    trajectories = load_trajectories([Path(p) for p in args.trajectories])
    if not summary_rows or not trajectories:
        raise SystemExit("missing summary or trajectory rows")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    milestone_rows: List[Dict[str, object]] = []
    for row in sorted(summary_rows, key=lambda r: (r["p"], r["weight_decay"], r["seed"])):
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        milestone_rows.append(
            {
                "p": int(row["p"]),
                "seed": int(row["seed"]),
                "weight_decay": float(row["weight_decay"]),
                "t_fit": epoch_or_blank(row["t_fit"]),
                "t10": epoch_or_blank(first_ge(traj, "test_acc", 0.10)),
                "t20": epoch_or_blank(first_ge(traj, "test_acc", 0.20)),
                "t50": epoch_or_blank(first_ge(traj, "test_acc", 0.50)),
                "t80": epoch_or_blank(first_ge(traj, "test_acc", 0.80)),
                "t90": epoch_or_blank(first_ge(traj, "test_acc", 0.90)),
                "t95": epoch_or_blank(first_ge(traj, "test_acc", 0.95)),
                "t_curv": epoch_or_blank(curvature_onset_epoch(traj)),
                "final_test_acc": float(row["final_test_acc"]),
                "final_test_loss": float(row["final_test_loss"]),
            }
        )
    write_csv(
        outdir / "grok_milestones.csv",
        ["p", "seed", "weight_decay", "t_fit", "t10", "t20", "t50", "t80", "t90", "t95", "t_curv", "final_test_acc", "final_test_loss"],
        milestone_rows,
    )

    fit_rows: List[Dict[str, object]] = []
    xs_wd = [float(row["weight_decay"]) for row in milestone_rows]
    ys_t95 = [float(row["t95"]) for row in milestone_rows if row["t95"] != ""]
    xs_wd_valid = [float(row["weight_decay"]) for row in milestone_rows if row["t95"] != ""]
    fit_specs: List[Tuple[str, Callable[[float], float], Callable[[float], float]]] = [
        ("t95 ~ wd", lambda wd: wd, lambda t: t),
        ("t95 ~ 1/wd", lambda wd: 1.0 / wd, lambda t: t),
        ("log t95 ~ wd", lambda wd: wd, lambda t: math.log(t)),
        ("log t95 ~ 1/wd", lambda wd: 1.0 / wd, lambda t: math.log(t)),
    ]
    for label, xf, yf in fit_specs:
        xs = [xf(wd) for wd in xs_wd_valid]
        ys = [yf(t95) for t95 in ys_t95]
        intercept, slope, r2 = fit_linear(xs, ys)
        fit_rows.append({"model": label, "intercept": intercept, "slope": slope, "r2": r2})
    fit_rows.append({"model": "alignment_mse_norm_t50", "intercept": "", "slope": "", "r2": normalized_alignment_mse(trajectories, milestone_rows, "t50")})
    fit_rows.append({"model": "alignment_mse_norm_t95", "intercept": "", "slope": "", "r2": normalized_alignment_mse(trajectories, milestone_rows, "t95")})
    write_csv(outdir / "grok_onset_fit_screen.csv", ["model", "intercept", "slope", "r2"], fit_rows)

    plot_raw(trajectories, "test_acc", outdir / "grok_test_acc_raw.png")
    plot_raw(trajectories, "test_loss", outdir / "grok_test_loss_raw.png")
    plot_normalized(trajectories, milestone_rows, "t50", outdir / "grok_test_acc_norm_t50.png")
    plot_normalized(trajectories, milestone_rows, "t95", outdir / "grok_test_acc_norm_t95.png")
    fit_shape_laws(trajectories, milestone_rows, outdir / "grok_gompertz_fit.csv", outdir / "grok_gompertz_norm_t50.png")
    fit_shifted_rising_logistic(
        trajectories,
        milestone_rows,
        outdir / "grok_rise_logistic_fit.csv",
        outdir / "grok_rise_logistic_norm_t50.png",
        shift_field="t10",
        model_name="rise_logistic_norm_t50_shift_t10",
        xlabel="(Epoch - t10) / t50",
        title="Rising-phase logistic fit after onset shift",
    )
    fit_shifted_rising_logistic(
        trajectories,
        milestone_rows,
        outdir / "grok_rise_logistic_t20_fit.csv",
        outdir / "grok_rise_logistic_t20_norm_t50.png",
        shift_field="t20",
        model_name="rise_logistic_norm_t50_shift_t20",
        xlabel="(Epoch - t20) / t50",
        title="Rising-phase logistic fit with t20 onset shift",
    )
    fit_shifted_rising_logistic(
        trajectories,
        milestone_rows,
        outdir / "grok_rise_logistic_curvature_fit.csv",
        outdir / "grok_rise_logistic_curvature_norm_t50.png",
        shift_field="t_curv",
        model_name="rise_logistic_norm_t50_shift_tcurv",
        xlabel="(Epoch - t_curv) / t50",
        title="Rising-phase logistic fit with curvature onset shift",
    )
    fit_rising_phase_loss_logistic(
        trajectories,
        milestone_rows,
        outdir / "grok_rise_loss_logistic_fit.csv",
        outdir / "grok_rise_loss_logistic_norm_t50.png",
    )
    fit_rising_phase_logistic_fitted_t0(
        trajectories,
        milestone_rows,
        outdir / "grok_rise_logistic_fitted_t0_fit.csv",
        outdir / "grok_rise_logistic_fitted_t0_norm_t50.png",
    )
    fit_rising_phase_logistic_fixed_ct50(
        trajectories,
        milestone_rows,
        outdir / "grok_rise_logistic_fixed_ct50_fit.csv",
        outdir / "grok_rise_logistic_fixed_ct50_norm_t50.png",
    )

    print(f"[ok] wrote {outdir / 'grok_milestones.csv'}")
    print(f"[ok] wrote {outdir / 'grok_onset_fit_screen.csv'}")
    print(f"[ok] wrote {outdir / 'grok_rise_logistic_fixed_ct50_fit.csv'}")


if __name__ == "__main__":
    main()
