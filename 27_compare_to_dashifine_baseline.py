from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_fit_metric(rows: List[Dict[str, str]], model: str, key: str) -> float:
    for row in rows:
        if row["model"] == model and row.get(key, "") != "":
            return float(row[key])
    return float("nan")


def load_local_metrics(analysis_dir: Path, summary_csv: Path, label: str, task: str, optimizer: str) -> Dict[str, object]:
    onset_rows = read_csv_rows(analysis_dir / "grok_onset_fit_screen.csv")
    fixed_path = analysis_dir / "grok_rise_logistic_fixed_ct50_fit.csv"
    fitted_path = analysis_dir / "grok_rise_logistic_fitted_t0_fit.csv"
    fixed_rows = read_csv_rows(fixed_path) if fixed_path.exists() else []
    fitted_rows = read_csv_rows(fitted_path) if fitted_path.exists() else []
    milestone_rows = read_csv_rows(analysis_dir / "grok_milestones.csv")
    summary_rows = read_csv_rows(summary_csv)

    mean_train = sum(float(row["final_train_acc"]) for row in summary_rows) / len(summary_rows)
    mean_test = sum(float(row["final_test_acc"]) for row in summary_rows) / len(summary_rows)
    tfit_vals = [float(row["t_fit"]) for row in milestone_rows if row["t_fit"]]
    t50_vals = [float(row["t50"]) for row in milestone_rows if row["t50"]]
    t95_vals = [float(row["t95"]) for row in milestone_rows if row["t95"]]
    mean_tfit = sum(tfit_vals) / len(tfit_vals) if tfit_vals else float("nan")
    mean_t50 = sum(t50_vals) / len(t50_vals) if t50_vals else float("nan")
    mean_t95 = sum(t95_vals) / len(t95_vals) if t95_vals else float("nan")

    return {
        "label": label,
        "task": task,
        "architecture": "translated_leech_modular_classifier",
        "optimizer": optimizer,
        "run_count": len(summary_rows),
        "milestone_count": len(milestone_rows),
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


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_predeclared_criterion(
    baseline: Dict[str, object],
    local: Dict[str, object],
    criterion: Dict[str, object] | None,
) -> Dict[str, object]:
    """Evaluate only an externally supplied, explicitly predeclared criterion.

    There are deliberately no built-in scientific thresholds here.  If no
    criterion file is supplied, or if it does not certify that it was declared
    before held-out evaluation, this function refuses family-level promotion.
    """
    receipt: Dict[str, object] = {
        "criterion_present": criterion is not None,
        "criterion_declared_before_heldout_evaluation": False,
        "checks": {},
        "heldout_criterion_pass": False,
        "family_claim_supported": False,
    }
    if criterion is None:
        return receipt

    declared = criterion.get("declared_before_heldout_evaluation") is True
    receipt["criterion_declared_before_heldout_evaluation"] = declared
    receipt["criterion_name"] = criterion.get("name", "unnamed")
    receipt["criterion_version"] = criterion.get("version", "unspecified")
    if not declared:
        return receipt

    checks: Dict[str, bool] = {}

    max_fixed_mse = criterion.get("max_fixed_shared_onset_logistic_mse")
    if max_fixed_mse is not None:
        value = local["fixed_shared_onset_logistic_mse"]
        checks["fixed_shared_onset_logistic_mse"] = _finite(value) and float(value) <= float(max_fixed_mse)

    max_onset_delta = criterion.get("max_abs_shared_onset_delta_from_baseline")
    if max_onset_delta is not None:
        local_c = local["shared_onset_c"]
        baseline_c = baseline["shared_onset_c"]
        checks["shared_onset_delta_from_baseline"] = (
            _finite(local_c)
            and _finite(baseline_c)
            and abs(float(local_c) - float(baseline_c)) <= float(max_onset_delta)
        )

    min_final_test_acc = criterion.get("min_final_test_acc")
    if min_final_test_acc is not None:
        value = local["final_test_acc"]
        checks["final_test_acc"] = _finite(value) and float(value) >= float(min_final_test_acc)

    min_runs = criterion.get("min_run_count")
    if min_runs is not None:
        checks["run_count"] = int(local["run_count"]) >= int(min_runs)

    receipt["checks"] = checks
    receipt["heldout_criterion_pass"] = bool(checks) and all(checks.values())
    receipt["family_claim_supported"] = bool(receipt["heldout_criterion_pass"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", default="phase2_validation/dashifine_shared_onset_logistic_baseline.json")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--task", default="mul")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--architecture", default="translated_leech_modular_classifier")
    parser.add_argument("--label", default="leech_phase2")
    parser.add_argument("--out-prefix", default="phase2_comparison")
    parser.add_argument(
        "--criterion-json",
        default=None,
        help="Optional externally predeclared held-out criterion. No default scientific thresholds are invented.",
    )
    args = parser.parse_args()

    with Path(args.baseline_json).open() as f:
        baseline = json.load(f)

    baseline_row = {
        "label": "dashifine_baseline",
        "task": baseline["task"],
        "architecture": "dashi_baseline",
        "optimizer": "adamw",
        "run_count": baseline.get("run_count", ""),
        "milestone_count": baseline.get("milestone_count", ""),
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
    local_row["architecture"] = args.architecture

    fields = list(baseline_row.keys())
    csv_path = Path(f"{args.out_prefix}.csv")
    md_path = Path(f"{args.out_prefix}.md")
    receipt_path = Path(f"{args.out_prefix}.receipt.json")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(baseline_row)
        writer.writerow(local_row)

    with md_path.open("w") as f:
        f.write("# Phase 2 Comparison\n\n")
        f.write("| label | architecture | optimizer | runs | shared_c | fixed_mse | fitted_mse | t95~1/wd r2 | align t50 | align t95 | final test acc |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in [baseline_row, local_row]:
            f.write(
                f"| {row['label']} | {row['architecture']} | {row['optimizer']} | {row['run_count']} | "
                f"{row['shared_onset_c']} | {row['fixed_shared_onset_logistic_mse']} | "
                f"{row['fitted_onset_logistic_mse']} | {row['t95_inverse_weight_decay_r2']} | "
                f"{row['alignment_mse_norm_t50']} | {row['alignment_mse_norm_t95']} | "
                f"{row['final_test_acc']} |\n"
            )
        f.write("\n")
        f.write(f"Baseline source note: `{baseline['source_note']}`\n")

    criterion = None
    if args.criterion_json is not None:
        with Path(args.criterion_json).open() as f:
            criterion = json.load(f)

    criterion_receipt = evaluate_predeclared_criterion(baseline_row, local_row, criterion)
    receipt = {
        "schema": "dashi.grokking.heldout-receipt.v1",
        "scope": "finite_empirical_comparison",
        "baseline_source": str(args.baseline_json),
        "analysis_dir": str(args.analysis_dir),
        "summary_csv": str(args.summary_csv),
        "task": args.task,
        "architecture": args.architecture,
        "optimizer": args.optimizer,
        "label": args.label,
        "run_count": local_row["run_count"],
        "milestone_count": local_row["milestone_count"],
        "metrics": {
            "shared_onset_c": local_row["shared_onset_c"],
            "fixed_shared_onset_logistic_mse": local_row["fixed_shared_onset_logistic_mse"],
            "fitted_onset_logistic_mse": local_row["fitted_onset_logistic_mse"],
            "t95_inverse_weight_decay_r2": local_row["t95_inverse_weight_decay_r2"],
            "alignment_mse_norm_t50": local_row["alignment_mse_norm_t50"],
            "alignment_mse_norm_t95": local_row["alignment_mse_norm_t95"],
            "final_test_acc": local_row["final_test_acc"],
        },
        "criterion_receipt": criterion_receipt,
        "exact_family_identity_claimed": False,
        "posthoc_tolerance_counts_as_independent_validation": False,
        "unsupported_runtime_failure_counts_as_model_semantics": False,
    }
    with receipt_path.open("w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {receipt_path}")


if __name__ == "__main__":
    main()
