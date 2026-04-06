#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASIN_LABELS = ("incoherent", "coherent_bad", "coherent_safe")


def robust_scale_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(x).all():
            raise ValueError(f"column {col!r} is empty after numeric coercion")
        good = ~np.isnan(x)
        fill = np.nanmedian(x[good])
        x = np.where(good, x, fill)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-12
        out[col] = np.tanh((x - med) / (1.4826 * mad))
    return out


@dataclass(frozen=True)
class Signature:
    p: int
    q: int
    z: int
    signs: Tuple[int, ...]

    @property
    def mask(self) -> str:
        return ",".join(str(s) for s in self.signs)


def all_signatures(dim: int, allow_zero: bool) -> List[Signature]:
    values = (-1, 0, 1) if allow_zero else (-1, 1)
    out: List[Signature] = []
    for signs in itertools.product(values, repeat=dim):
        p = sum(1 for s in signs if s == 1)
        q = sum(1 for s in signs if s == -1)
        z = sum(1 for s in signs if s == 0)
        if p + q == 0:
            continue
        out.append(Signature(p=p, q=q, z=z, signs=tuple(signs)))
    return out


def quadratic_delta(dx: np.ndarray, signs: Sequence[int]) -> np.ndarray:
    sign_array = np.asarray(signs, dtype=float)
    return ((dx ** 2) * sign_array[None, :]).sum(axis=1)


def compute_forward_mask(arrow_delta: np.ndarray, eps_arrow: float, mode: str) -> np.ndarray:
    if mode == "decreasing":
        return arrow_delta <= eps_arrow
    if mode == "increasing":
        return arrow_delta >= -eps_arrow
    raise ValueError(f"unknown arrow mode: {mode}")


def score_signature(
    dx: np.ndarray,
    arrow_delta: np.ndarray,
    signs: Sequence[int],
    eps_cone: float,
    eps_arrow: float,
    arrow_mode: str,
) -> Dict[str, float]:
    qd = quadratic_delta(dx, signs)
    cone_ok = qd <= eps_cone
    forward = compute_forward_mask(arrow_delta, eps_arrow, arrow_mode)
    both = cone_ok & forward
    forward_count = int(forward.sum())
    return {
        "cone_frac": float(cone_ok.mean()),
        "forward_frac": float(forward.mean()),
        "forward_cone_frac_all": float(both.mean()),
        "forward_cone_frac_conditional": float(cone_ok[forward].mean()) if forward_count else float("nan"),
        "n_steps": int(len(qd)),
        "n_forward": forward_count,
        "n_cone": int(cone_ok.sum()),
        "mean_Qd": float(np.mean(qd)),
        "min_Qd": float(np.min(qd)),
        "max_Qd": float(np.max(qd)),
    }


def _optional_numeric(df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def derive_basin_labels(
    df: pd.DataFrame,
    coherence_col: str,
    bad_mode_col: str,
    coherence_thresh: float,
    bad_mode_thresh: float,
) -> List[str]:
    coherence = _optional_numeric(df, coherence_col)
    bad_mode = _optional_numeric(df, bad_mode_col)
    if coherence is None or bad_mode is None:
        return []

    labels: List[str] = []
    for c, b in zip(coherence, bad_mode):
        if np.isnan(c) or c < coherence_thresh:
            labels.append("incoherent")
        elif not np.isnan(b) and b >= bad_mode_thresh:
            labels.append("coherent_bad")
        else:
            labels.append("coherent_safe")
    return labels


def resolve_basin_labels(
    df: pd.DataFrame,
    basin_col: Optional[str],
    coherence_col: str,
    bad_mode_col: str,
    coherence_thresh: float,
    bad_mode_thresh: float,
) -> List[str]:
    if basin_col and basin_col in df.columns:
        labels = [str(v).strip() for v in df[basin_col].fillna("")]
        return [label if label in BASIN_LABELS else "incoherent" for label in labels]
    return derive_basin_labels(df, coherence_col, bad_mode_col, coherence_thresh, bad_mode_thresh)


def _run_lengths(basins: Sequence[str], target: str) -> List[int]:
    out: List[int] = []
    run = 0
    for basin in basins:
        if basin == target:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def regime_stats(basins: Sequence[str]) -> Dict[str, float]:
    n = len(basins)
    if n == 0:
        return {}

    bad = [b == "coherent_bad" for b in basins]
    safe = [b == "coherent_safe" for b in basins]
    incoherent = [b == "incoherent" for b in basins]
    bad_runs = _run_lengths(basins, "coherent_bad")
    safe_runs = _run_lengths(basins, "coherent_safe")

    bad_to_safe = 0
    safe_to_bad = 0
    suppression_steps = 0
    for a, b in zip(basins[:-1], basins[1:]):
        if a == "coherent_bad" and b == "coherent_safe":
            bad_to_safe += 1
            suppression_steps += 1
        if a == "coherent_safe" and b == "coherent_bad":
            safe_to_bad += 1

    bad_count = sum(bad)
    safe_count = sum(safe)
    return {
        "incoherent_dwell_frac": sum(incoherent) / n,
        "bad_dwell_frac": bad_count / n,
        "safe_dwell_frac": safe_count / n,
        "bad_to_safe_ratio": bad_count / max(safe_count, 1),
        "n_bad_to_safe_transitions": bad_to_safe,
        "n_safe_to_bad_transitions": safe_to_bad,
        "mean_bad_run_length": float(np.mean(bad_runs)) if bad_runs else 0.0,
        "mean_safe_run_length": float(np.mean(safe_runs)) if safe_runs else 0.0,
        "n_bad_suppression_steps": suppression_steps,
    }


def write_basin_transitions(path: str, steps: Sequence[int], basins: Sequence[str]) -> None:
    rows: List[Dict[str, object]] = []
    for i in range(len(basins) - 1):
        rows.append(
            {
                "step_from": int(steps[i]),
                "step_to": int(steps[i + 1]),
                "basin_from": basins[i],
                "basin_to": basins[i + 1],
                "bad_suppression_step": int(basins[i] == "coherent_bad" and basins[i + 1] == "coherent_safe"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DASHI-style delta-cone scan on external LILA logs.")
    parser.add_argument("--csv", required=True, help="CSV log with step and feature columns")
    parser.add_argument("--features", required=True, help="Comma-separated feature columns")
    parser.add_argument("--arrow", required=True, help="Arrow column used only for forward filtering")
    parser.add_argument("--step", default="step", help="Step column")
    parser.add_argument("--eps", type=float, default=0.15, help="Cone threshold")
    parser.add_argument("--eps-arrow", type=float, default=1e-9, help="Arrow tolerance")
    parser.add_argument("--arrow-mode", choices=["decreasing", "increasing"], default="decreasing")
    parser.add_argument("--allow-zero", action="store_true", help="Allow zero-sign coordinates in the signature family")
    parser.add_argument("--require-nondegenerate", action="store_true", help="Keep only signatures with z = 0")
    parser.add_argument("--basin-col", help="Optional basin label column")
    parser.add_argument("--coherence-col", default="coherence_score", help="Coherence score column used when basin labels are derived")
    parser.add_argument("--bad-mode-col", default="bad_mode_score", help="Bad-mode score column used when basin labels are derived")
    parser.add_argument("--coherence-thresh", type=float, default=0.6, help="Threshold for coherence gating")
    parser.add_argument("--bad-mode-thresh", type=float, default=0.5, help="Threshold for coherent_bad classification")
    parser.add_argument("--regime-out", help="Optional JSON output for bad-mode suppression summary")
    parser.add_argument("--basin-transitions-out", help="Optional CSV output for basin-to-basin transitions")
    parser.add_argument("--out", required=True, help="Output ranking CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values(args.step).reset_index(drop=True)
    features = [item.strip() for item in args.features.split(",") if item.strip()]
    if args.arrow not in features:
        raise ValueError("--arrow must be included in --features")
    missing = [col for col in [args.step, *features] if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    scaled = robust_scale_frame(df, features)
    q_cols = [col for col in features if col != args.arrow]
    if not q_cols:
        raise ValueError("need at least one non-arrow feature")

    x = scaled[q_cols].to_numpy(dtype=float)
    arrow = scaled[args.arrow].to_numpy(dtype=float)
    dx = x[1:] - x[:-1]
    darrow = arrow[1:] - arrow[:-1]

    rows: List[Dict[str, object]] = []
    for signature in all_signatures(len(q_cols), args.allow_zero):
        if args.require_nondegenerate and signature.z != 0:
            continue
        rows.append(
            {
                "p": signature.p,
                "q": signature.q,
                "z": signature.z,
                "mask": signature.mask,
                **score_signature(dx, darrow, signature.signs, args.eps, args.eps_arrow, args.arrow_mode),
            }
        )

    rank = pd.DataFrame(rows).sort_values(
        by=[
            "forward_cone_frac_conditional",
            "forward_frac",
            "cone_frac",
            "z",
            "mean_Qd",
        ],
        ascending=[False, False, False, True, True],
    )
    rank.to_csv(args.out, index=False)
    print(f"saved: {args.out}")
    print(f"quadratic columns: {q_cols}")

    basins = resolve_basin_labels(
        df,
        args.basin_col,
        args.coherence_col,
        args.bad_mode_col,
        args.coherence_thresh,
        args.bad_mode_thresh,
    )
    if basins:
        summary = regime_stats(basins)
        print("=== Bad-mode suppression ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        if args.regime_out:
            with open(args.regime_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"saved: {args.regime_out}")
        if args.basin_transitions_out:
            write_basin_transitions(args.basin_transitions_out, df[args.step].tolist(), basins)
            print(f"saved: {args.basin_transitions_out}")


if __name__ == "__main__":
    main()
