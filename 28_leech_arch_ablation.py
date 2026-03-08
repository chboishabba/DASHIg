from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_WEIGHT_DECAYS = [0.20, 0.22, 0.24, 0.25, 0.30, 0.35, 0.40]
DEFAULT_LAMBDAS = [0.0, 0.001, 0.01]


def slugify_lambda(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "neg").replace(".", "p")


def run(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def read_first_row(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows[0]


def read_last_row(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows[-1]


def output_exists(scan_prefix: Path, analysis_dir: Path, comparison_prefix: Path) -> bool:
    return (
        Path(str(scan_prefix) + ".csv").exists()
        and Path(str(scan_prefix) + "_trajectories.csv").exists()
        and (analysis_dir / "grok_onset_fit_screen.csv").exists()
        and Path(str(comparison_prefix) + ".csv").exists()
        and Path(str(comparison_prefix) + ".md").exists()
    )


def collect_summary_row(
    *,
    label: str,
    lambda_geo: float,
    task: str,
    optimizer: str,
    scan_prefix: Path,
    analysis_dir: Path,
    comparison_prefix: Path,
) -> Dict[str, object]:
    comparison_row = read_last_row(Path(str(comparison_prefix) + ".csv"))
    scan_row = read_first_row(Path(str(scan_prefix) + ".csv"))
    return {
        "label": label,
        "lambda_geo": lambda_geo,
        "task": task,
        "optimizer": optimizer,
        "summary_csv": str(scan_prefix) + ".csv",
        "analysis_dir": str(analysis_dir),
        "comparison_csv": str(comparison_prefix) + ".csv",
        "comparison_md": str(comparison_prefix) + ".md",
        "mean_t50": comparison_row["t50"],
        "mean_t95": comparison_row["t95"],
        "shared_onset_c": comparison_row["shared_onset_c"],
        "fixed_shared_onset_logistic_mse": comparison_row["fixed_shared_onset_logistic_mse"],
        "fitted_onset_logistic_mse": comparison_row["fitted_onset_logistic_mse"],
        "t95_inverse_weight_decay_r2": comparison_row["t95_inverse_weight_decay_r2"],
        "alignment_mse_norm_t50": comparison_row["alignment_mse_norm_t50"],
        "final_test_acc": comparison_row["final_test_acc"],
        "stop_epoch": scan_row["stop_epoch"],
    }


def write_summary(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    fieldnames = [
        "label",
        "lambda_geo",
        "task",
        "optimizer",
        "summary_csv",
        "analysis_dir",
        "comparison_csv",
        "comparison_md",
        "mean_t50",
        "mean_t95",
        "shared_onset_c",
        "fixed_shared_onset_logistic_mse",
        "fitted_onset_logistic_mse",
        "t95_inverse_weight_decay_r2",
        "alignment_mse_norm_t50",
        "final_test_acc",
        "stop_epoch",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as f:
        f.write("# Leech Architecture Ablation Runs\n\n")
        f.write("| label | lambda_geo | shared_c | fixed_mse | fitted_mse | t95~1/wd r2 | align t50 | final test acc | comparison |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['label']} | {row['lambda_geo']} | {row['shared_onset_c']} | "
                f"{row['fixed_shared_onset_logistic_mse']} | {row['fitted_onset_logistic_mse']} | "
                f"{row['t95_inverse_weight_decay_r2']} | {row['alignment_mse_norm_t50']} | "
                f"{row['final_test_acc']} | "
                f"`{row['comparison_csv']}` |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the translated Leech architecture ablation ladder without changing "
            "the existing benchmark scripts. This script orchestrates the scan, "
            "analysis, and baseline comparison for each lambda_geo setting."
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
    parser.add_argument("--lambda-geos", type=float, nargs="+", default=DEFAULT_LAMBDAS)
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-root", type=str, default="leech_arch_ablation_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []

    for lambda_geo in args.lambda_geos:
        label = f"{args.task}_{args.optimizer}_lambda_{slugify_lambda(lambda_geo)}"
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)

        scan_prefix = run_dir / "scan"
        analysis_dir = run_dir / "analysis"
        comparison_prefix = run_dir / "comparison"

        if args.skip_existing and output_exists(scan_prefix, analysis_dir, comparison_prefix):
            print(f"Skipping existing run: {label}")
        else:
            scan_cmd = [
                sys.executable,
                "27_leech_grok_critical_scan.py",
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
                "--lambda-geo",
                str(lambda_geo),
                "--dropout",
                str(args.dropout),
                "--torch-device",
                args.torch_device,
                "--out-prefix",
                str(scan_prefix),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--weight-decays",
                *[str(wd) for wd in args.weight_decays],
            ]
            run(scan_cmd, cwd)

            analysis_cmd = [
                sys.executable,
                "27_leech_trajectory_analysis.py",
                "--summary",
                str(scan_prefix) + ".csv",
                "--trajectories",
                str(scan_prefix) + "_trajectories.csv",
                "--outdir",
                str(analysis_dir),
            ]
            run(analysis_cmd, cwd)

            comparison_cmd = [
                sys.executable,
                "27_compare_to_dashifine_baseline.py",
                "--analysis-dir",
                str(analysis_dir),
                "--summary-csv",
                str(scan_prefix) + ".csv",
                "--task",
                args.task,
                "--optimizer",
                args.optimizer,
                "--label",
                label,
                "--out-prefix",
                str(comparison_prefix),
            ]
            run(comparison_cmd, cwd)

        summary_rows.append(
            collect_summary_row(
                label=label,
                lambda_geo=lambda_geo,
                task=args.task,
                optimizer=args.optimizer,
                scan_prefix=scan_prefix,
                analysis_dir=analysis_dir,
                comparison_prefix=comparison_prefix,
            )
        )

    write_summary(
        summary_rows,
        output_root / "ablation_runs.csv",
        output_root / "ablation_runs.md",
    )
    print(f"Saved: {output_root / 'ablation_runs.csv'}")
    print(f"Saved: {output_root / 'ablation_runs.md'}")


if __name__ == "__main__":
    main()
