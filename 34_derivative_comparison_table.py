from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_one_row(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows[0]


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w") as f:
        f.write("# Derivative Comparison Table\n\n")
        f.write("| label | mean_peak_x | mean_slope_proxy_k | mean_half_max_width | mean_pre_t50_area | mean_pre_t50_fraction | mean_gaussian_mse | mean_corr_to_mean | n_runs |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['label']} | {row['mean_peak_x']} | {row['mean_slope_proxy_k']} | "
                f"{row['mean_half_max_width']} | {row['mean_pre_t50_area']} | {row['mean_pre_t50_fraction']} | "
                f"{row['mean_gaussian_mse']} | {row['mean_corr_to_mean']} | {row['n_runs']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--out-prefix", default="derivative_comparison")
    args = parser.parse_args()

    rows = [read_one_row(Path(path)) for path in args.summaries]
    fieldnames = [
        "label",
        "mean_peak_x",
        "std_peak_x",
        "mean_slope_proxy_k",
        "mean_half_max_width",
        "mean_pre_t50_area",
        "mean_pre_t50_fraction",
        "mean_skew",
        "mean_gaussian_mse",
        "mean_corr_to_mean",
        "n_runs",
    ]
    csv_path = Path(f"{args.out_prefix}.csv")
    md_path = Path(f"{args.out_prefix}.md")
    write_csv(csv_path, fieldnames, rows)
    write_md(md_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
