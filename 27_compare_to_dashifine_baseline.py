from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_fit_metric(rows: List[Dict[str, str]], model: str, key: str) -> float:
    for row in rows:
        if row["model"] == model and row.get(key, "") != "":
            return float(row[key])
    raise KeyError(f"missing {model}:{key}")


def load_local_metrics(analysis_dir: Path, summary_csv: Path, label: str, task: str, optimizer: str) -> Dict[str, object]:
    onset_rows = read_csv_rows(analysis_dir / "grok_onset_fit_screen.csv")
    fixed_rows = read_csv_rows(analysis_dir / "grok_rise_logistic_fixed_ct50_fit.csv")
    fitted_rows = read_csv_rows(analysis_dir / "grok_rise_logistic_fitted_t0_fit.csv")
    milestone_rows = read_csv_rows(analysis_dir / "grok_milestones.csv")
    summary_rows = read_csv_rows(summary_csv)

    mean_train = sum(float(row["final_train_acc"]) for row in summary_rows) / len(summary_rows)
    mean_test = sum(float(row["final_test_acc"]) for row in summary_rows) / len(summary_rows)
    mean_tfit = sum(float(row["t_fit"]) for row in milestone_rows if row["t_fit"]) / max(1, sum(1 for row in milestone_rows if row["t_fit"]))
    mean_t50 = sum(float(row["t50"]) for row in milestone_rows) / len(milestone_rows)
    mean_t95 = sum(float(row["t95"]) for row in milestone_rows) / len(milestone_rows)

    return {
        "label": label,
        "task": task,
        "architecture": "translated_leech_modular_classifier",
        "optimizer": optimizer,
        "t_fit": mean_tfit,
        "t50": mean_t50,
        "t95": mean_t95,
        "shared_onset_c": read_fit_metric(fixed_rows, "shared_c", "param_1"),
        "fixed_shared_onset_logistic_mse": read_fit_metric(fixed_rows, "rise_logistic_norm_t50_fixed_ct50", "mse"),
        "fitted_onset_logistic_mse": read_fit_metric(fitted_rows, "rise_logistic_norm_t50_fitted_t0", "mse"),
        "t95_inverse_weight_decay_r2": read_fit_metric(onset_rows, "t95 ~ 1/wd", "r2"),
        "alignment_mse_norm_t50": read_fit_metric(onset_rows, "alignment_mse_norm_t50", "r2"),
        "alignment_mse_norm_t95": read_fit_metric(onset_rows, "alignment_mse_norm_t95", "r2"),
        "final_train_acc": mean_train,
        "final_test_acc": mean_test,
        "trajectory_note": "Generated from local Phase 2 harness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", default="phase2_validation/dashifine_shared_onset_logistic_baseline.json")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--task", default="mul")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--label", default="leech_phase2")
    parser.add_argument("--out-prefix", default="phase2_comparison")
    args = parser.parse_args()

    with Path(args.baseline_json).open() as f:
        baseline = json.load(f)

    baseline_row = {
        "label": "dashifine_baseline",
        "task": baseline["task"],
        "architecture": "dashi_baseline",
        "optimizer": "adamw",
        "t_fit": "",
        "t50": "",
        "t95": "",
        "shared_onset_c": baseline["metrics"]["fixed_shared_onset_logistic"]["shared_c"],
        "fixed_shared_onset_logistic_mse": baseline["metrics"]["fixed_shared_onset_logistic"]["mse"],
        "fitted_onset_logistic_mse": baseline["metrics"]["fitted_onset_logistic"]["mse"],
        "t95_inverse_weight_decay_r2": baseline["metrics"]["t95_inverse_weight_decay_r2"],
        "alignment_mse_norm_t50": baseline["metrics"]["alignment_mse_norm_t50"],
        "alignment_mse_norm_t95": baseline["metrics"]["alignment_mse_norm_t95"],
        "final_train_acc": "",
        "final_test_acc": "",
        "trajectory_note": baseline["trajectory_note"],
    }
    local_row = load_local_metrics(Path(args.analysis_dir), Path(args.summary_csv), args.label, args.task, args.optimizer)

    fields = list(baseline_row.keys())
    csv_path = Path(f"{args.out_prefix}.csv")
    md_path = Path(f"{args.out_prefix}.md")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(baseline_row)
        writer.writerow(local_row)

    with md_path.open("w") as f:
        f.write("# Phase 2 Comparison\n\n")
        f.write("| label | architecture | optimizer | shared_c | fixed_mse | fitted_mse | t95~1/wd r2 | align t50 | align t95 | final test acc |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in [baseline_row, local_row]:
            f.write(
                f"| {row['label']} | {row['architecture']} | {row['optimizer']} | "
                f"{row['shared_onset_c']} | {row['fixed_shared_onset_logistic_mse']} | "
                f"{row['fitted_onset_logistic_mse']} | {row['t95_inverse_weight_decay_r2']} | "
                f"{row['alignment_mse_norm_t50']} | {row['alignment_mse_norm_t95']} | "
                f"{row['final_test_acc']} |\n"
            )
        f.write("\n")
        f.write(f"Baseline source note: `{baseline['source_note']}`\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
