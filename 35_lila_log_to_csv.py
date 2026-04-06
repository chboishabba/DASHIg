from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


STEP_PATTERN = re.compile(r"📊 Шаг\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)")
VAL_PATTERN = re.compile(r"Validation Loss:\s+([\d.]+)")


SR_CN_ALIASES = {
    "sr_block0": [
        "sr_block0",
        "blocks.0.attn.qkv.weight_sr",
    ],
    "sr_block5": [
        "sr_block5",
        "blocks.5.attn.qkv.weight_sr",
    ],
    "sr_block11": [
        "sr_block11",
        "blocks.11.attn.qkv.weight_sr",
    ],
    "cn_block0": [
        "cn_block0",
        "blocks.0.attn.qkv.weight_cn",
    ],
    "cn_block5": [
        "cn_block5",
        "blocks.5.attn.qkv.weight_cn",
    ],
    "cn_block11": [
        "cn_block11",
        "blocks.11.attn.qkv.weight_cn",
    ],
}


def parse_log(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        step_match = STEP_PATTERN.search(line)
        if not step_match:
            i += 1
            continue

        step = int(step_match.group(1))
        train_loss = float(step_match.group(2))
        val_loss: Optional[float] = None

        for offset in range(1, 4):
            if i + offset >= len(lines):
                break
            val_match = VAL_PATTERN.search(lines[i + offset].strip())
            if val_match:
                val_loss = float(val_match.group(1))
                i += offset
                break

        rows.append(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": "" if val_loss is None else val_loss,
            }
        )
        i += 1
    return rows


def _pick_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def merge_sr_csv(base_df: pd.DataFrame, sr_csv: Path) -> pd.DataFrame:
    sr_df = pd.read_csv(sr_csv)
    if "step" not in sr_df.columns:
        raise ValueError(f"stable-rank CSV missing 'step': {sr_csv}")

    keep = ["step"]
    rename: Dict[str, str] = {}
    for target, aliases in SR_CN_ALIASES.items():
        picked = _pick_column(sr_df, aliases)
        if picked is not None:
            keep.append(picked)
            rename[picked] = target

    merged = sr_df[keep].rename(columns=rename)
    return base_df.merge(merged, on="step", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Leech/LILA markdown logs into CSV.")
    parser.add_argument("--log", required=True, help="Input markdown log from LeechTransformer/train_logs")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--model-name", default="lila_external", help="Model label written into the CSV")
    parser.add_argument(
        "--sr-csv",
        help="Optional stable-rank / condition-number CSV to align on step. Supports notebook column names.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no training rows parsed from {log_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if args.sr_csv:
        df = merge_sr_csv(df, Path(args.sr_csv))

    base_fields = ["model_name", "source_log", "step", "train_loss", "val_loss"]
    extra_fields = [
        field
        for field in ["sr_block0", "sr_block5", "sr_block11", "cn_block0", "cn_block5", "cn_block11"]
        if field in df.columns
    ]
    fields = [*base_fields, *extra_fields]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in df.to_dict(orient="records"):
            writer.writerow(
                {
                    "model_name": args.model_name,
                    "source_log": str(log_path),
                    **row,
                }
            )

    print(f"saved: {out_path}")
    print(f"rows: {len(df)}")
    if extra_fields:
        print(f"aligned representation columns: {extra_fields}")


if __name__ == "__main__":
    main()
