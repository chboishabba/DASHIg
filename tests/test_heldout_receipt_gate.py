from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "27_compare_to_dashifine_baseline.py"
spec = importlib.util.spec_from_file_location("phase2_compare", MODULE_PATH)
assert spec is not None and spec.loader is not None
phase2_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2_compare)


def _baseline() -> dict[str, object]:
    return {
        "shared_onset_c": 0.80,
        "fixed_shared_onset_logistic_mse": 0.001,
    }


def _local() -> dict[str, object]:
    return {
        "shared_onset_c": 0.79,
        "fixed_shared_onset_logistic_mse": 0.002,
        "final_test_acc": 0.99,
        "run_count": 8,
    }


def test_no_criterion_never_promotes_family_claim() -> None:
    receipt = phase2_compare.evaluate_predeclared_criterion(_baseline(), _local(), None)
    assert receipt["criterion_present"] is False
    assert receipt["heldout_criterion_pass"] is False
    assert receipt["family_claim_supported"] is False


def test_posthoc_criterion_never_promotes_family_claim() -> None:
    criterion = {
        "declared_before_heldout_evaluation": False,
        "max_fixed_shared_onset_logistic_mse": 1.0,
        "max_abs_shared_onset_delta_from_baseline": 1.0,
        "min_final_test_acc": 0.0,
        "min_run_count": 1,
    }
    receipt = phase2_compare.evaluate_predeclared_criterion(_baseline(), _local(), criterion)
    assert receipt["criterion_present"] is True
    assert receipt["criterion_declared_before_heldout_evaluation"] is False
    assert receipt["heldout_criterion_pass"] is False
    assert receipt["family_claim_supported"] is False


def test_predeclared_criterion_can_pass_only_when_all_declared_checks_pass() -> None:
    criterion = {
        "name": "regression fixture",
        "version": 1,
        "declared_before_heldout_evaluation": True,
        "max_fixed_shared_onset_logistic_mse": 0.01,
        "max_abs_shared_onset_delta_from_baseline": 0.02,
        "min_final_test_acc": 0.95,
        "min_run_count": 4,
    }
    receipt = phase2_compare.evaluate_predeclared_criterion(_baseline(), _local(), criterion)
    assert receipt["criterion_declared_before_heldout_evaluation"] is True
    assert receipt["heldout_criterion_pass"] is True
    assert receipt["family_claim_supported"] is True
    assert all(receipt["checks"].values())


def test_predeclared_criterion_fails_if_one_check_fails() -> None:
    criterion = {
        "declared_before_heldout_evaluation": True,
        "max_fixed_shared_onset_logistic_mse": 0.001,
        "min_run_count": 4,
    }
    receipt = phase2_compare.evaluate_predeclared_criterion(_baseline(), _local(), criterion)
    assert receipt["checks"]["fixed_shared_onset_logistic_mse"] is False
    assert receipt["heldout_criterion_pass"] is False
    assert receipt["family_claim_supported"] is False
