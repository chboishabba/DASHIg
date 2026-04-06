#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    main()
