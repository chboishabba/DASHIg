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
- Updated the translated Leech benchmark runner to expose configurable early stopping (`--grok-thr`, `--grok-patience-logs`) and default to stopping once test accuracy stays at or above `0.97` for 5 logged checkpoints, instead of always running to the maximum epoch budget.
- Refreshed the canonical repo context to the fetched `Grokking Valuation Resolution` thread (`6958b536-7e18-8320-bce9-421436b4ccf2` / `a450d4cb4d0be34146aab4df6898149e2910b472`).
- Updated the planning docs to record the current Leech Phase 2 readout as partial transfer rather than simple failure: logistic still fits better than Gompertz, but the shared onset shifts strongly earlier and the inverse-weight-decay timing screen becomes noisy.
- Reordered the immediate Phase 2 roadmap so Leech architecture ablations (`lambda_geo` ladder) come before second-optimizer and related-task validation.
