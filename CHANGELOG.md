# Changelog

## 2026-03-08

- Updated the canonical repo context from the older DASHI comparison thread to the fetched `Theorem Assistance` thread (`6958ff8a-03c8-8321-b906-30e48e412a3a` / `2aec04871d54bda5059cd98155cd7512f13ab503`).
- Added `GROKKING_TIME_RESCALING_NOTE.md` to document the current flagship empirical law: `t50` time-rescaling, shared onset `t0 ≈ 0.8055 * t50`, and shared post-onset logistic growth.
- Retargeted `TODO.md` toward local reproduction of the `t50`-based fits and away from treating the older DASHI formalism thread as the immediate active plan.
- Recorded the distinction between existing repo-local grokking scripts and the fit artifacts that are still external to this repo snapshot.
- Added `ROADMAP.md` to define the whole-program ML roadmap across compression-first learning, architectural priors, and observer/entropy analysis.
- Reframed the planning docs so the grokking-law note is Phase 1 of the broader ML program, with external validation as the next highest-value scientific step after local reproduction.
- Clarified ownership split: `../dashifine` owns the Phase 1 grokking-law baseline, this repo owns Phase 2 validation/comparison, and `../dashi_agda` owns the broader formalism track.
- Refocused this repo’s primary job on translating `LeechTransformer/` into a defensible second-architecture comparison against the DASHI baseline rather than re-owning baseline reproduction.
- Added `phase2_validation/` with a manual `../dashifine` baseline reference and the shared Phase 2 comparison contract.
- Added `27_leech_grok_critical_scan.py` plus `phase2_validation/leech_modular_benchmark.py` to run a translated Leech modular classifier benchmark with AdamW or SGD, modular multiplication or addition, resonance loss, and trajectory logging.
- Added `27_leech_trajectory_analysis.py` to reproduce the milestone/fitting surface locally for translated Leech runs, including `t50`/`t95` alignment and shared-onset logistic fits.
- Added `27_compare_to_dashifine_baseline.py` to generate local CSV/Markdown comparisons against the manually recorded DASHI baseline.
- Added `28_leech_arch_ablation.py` to orchestrate the Leech `lambda_geo` ablation ladder, run scan/analysis/comparison for each leg, and emit a top-level ablation summary.
- Added `29_leech_arch_ablation_prelim.py` as a shorter one-leg ablation entrypoint for representative-band checks before running the full ladder.
- Added `phase2_validation/plain_modular_benchmark.py`, `30_plain_grok_critical_scan.py`, and `31_plain_baseline_prelim.py` to provide a neutral plain-transformer modular baseline on the same scan/analysis surface as the translated Leech runs.
- Updated `27_compare_to_dashifine_baseline.py` to accept an explicit `--architecture` label so standard-baseline and Leech comparisons are reported correctly.
- Updated the translated Leech benchmark runner to expose configurable early stopping (`--grok-thr`, `--grok-patience-logs`) and default to stopping once test accuracy stays at or above `0.97` for 5 logged checkpoints, instead of always running to the maximum epoch budget.
- Refreshed the canonical repo context to the fetched `Grokking Valuation Resolution` thread (`6958b536-7e18-8320-bce9-421436b4ccf2` / `a450d4cb4d0be34146aab4df6898149e2910b472`).
- Updated the planning docs to record the current Leech Phase 2 readout as partial transfer rather than simple failure: logistic still fits better than Gompertz, but the shared onset shifts strongly earlier and the inverse-weight-decay timing screen becomes noisy.
- Reordered the immediate Phase 2 roadmap so Leech architecture ablations (`lambda_geo` ladder) come before second-optimizer and related-task validation.
- Added a neutral industry-standard modular baseline to the Phase 2 plan so Leech-vs-DASHI speed comparisons can be checked against a standard architecture before outreach.
- Recorded the first representative-band Leech ablation result (`lambda_geo = 0`, `wd = 0.22, 0.30`): both runs still grok cleanly, but the shared onset only shifts to `c ≈ 0.389`, so removing the geometric penalty alone does not restore the DASHI timing law.
- Reframed the immediate next step as finishing the representative-band prelims (`lambda_geo = 1e-3` and plain standard baseline) before deciding whether the full overnight ladder is worth the cost.
- Recorded the neutral plain-transformer prelim on the same representative band (`wd = 0.22, 0.30`): it also groks cleanly, has `shared_c ≈ 0.340`, and is competitive with the `lambda_geo = 0` Leech prelim rather than obviously dominated by it.
- Narrowed the next decision point again: run the `lambda_geo = 1e-3` Leech prelim, then compare all representative-band readouts before making strong outreach or speedup claims.
