# Context

This file captures the canonical context for the ChatGPT thread:

- Title: Grokking Valuation Resolution
- Conversation ID: 6958b536-7e18-8320-bce9-421436b4ccf2
- Canonical thread ID: a450d4cb4d0be34146aab4df6898149e2910b472
- Fetched via: direct online UUID pull into local structurer DB, then DB-first resolver lookup
- Date captured: 2026-03-08
- Storage status: persisted in `~/chat_archive.sqlite`

## Summary

The current canonical thread is focused on the grokking experiments in this repo, not the earlier DASHI-vs-E8 comparison thread. The thread refines how we should interpret the translated Leech Phase 2 result against the accepted `../dashifine` baseline.

Accepted upstream baseline:

- Test-accuracy trajectories collapse when time is rescaled by `t50`.
- The onset of rapid generalization is well approximated by one shared normalized location:
  - `t0 ≈ 0.8055 * t50`
- After that onset shift, the rise is well fit by a shared logistic curve.
- The fixed-onset and per-run-onset fits have nearly identical error:
  - `grok_rise_logistic_fixed_ct50_fit.csv`: `c ≈ 0.8055`, `mse ≈ 0.000360`
  - `grok_rise_logistic_fitted_t0_fit.csv`: `mse ≈ 0.000351`

Phase 2 Leech update from the fetched thread:

- The translated Leech model still appears better described by a logistic rise than by Gompertz.
- The current Leech fit does not preserve the baseline shared onset:
  - baseline `shared_c ≈ 0.8055`
  - translated Leech `shared_c ≈ 0.3254`
- The weight-decay timing screen is noisy rather than cleanly absent:
  - baseline `t95 ~ 1 / wd` `R^2 ≈ 0.9976`
  - translated Leech `t95 ~ 1 / wd` `R^2 ≈ 0.3866`
- The working interpretation is therefore not "the law vanished" but "the observation channel / basis changed the apparent onset and degraded the clean collapse."

The thread treats this as a valuation-resolution step: the Leech result is scientifically meaningful even though it does not match the baseline law cleanly. The immediate question is now whether this is structural or an artifact of the current translated prior / prime-base representation.

At repo level, this now sits inside a broader machine-learning comparison program captured in `ROADMAP.md`. The overall agenda is to compare:

- compression-first structure discovery (`DASHI` / grokking work at repo root)
- architectural-prior structure injection (`LeechTransformer/`)
- observer/entropy geometry analysis (`cognitive-observer-simulation/`)

Ownership split:

- `../dashifine` owns the upstream Phase 1 baseline for the grokking law
- this repo owns the external-validation and comparison work
- `../dashi_agda` owns the broader DASHI formalism track

## Flagship Empirical Law

For weight decay `lambda` in the near-critical regime:

`A_test(t; lambda) ≈ F(t / t50(lambda))`

with a shared normalized onset

`t0 / t50 ≈ 0.8055`

and a shared post-onset logistic rise.

Interpretation:

- `t50` is the natural clock of the late generalization dynamics.
- Weight decay mainly rescales time rather than changing the curve family.
- The onset of the escape into generalization occurs at a nearly fixed fraction of that clock.

## Repo-Level ML Goal

The broad ML goal in this repo is to compare different routes to structure and symmetry in learning systems:

- discover it through compression-driven dynamics
- inject it as an architectural prior
- analyze it through entropy/observer geometry

The active grokking-law thread is therefore the current benchmark layer, not the entire project. For this repo, it is an upstream input into Phase 2 rather than locally owned work.

## Latest Priorities

The latest user-side update in the fetched thread states that the best next step is:

1. Fold the empirical law directly into `GROKKING_TIME_RESCALING_NOTE.md`.
2. Treat that note as the flagship documentation artifact for the result.
3. Continue with mechanism work only after the descriptive law is written down cleanly.
4. After local reproduction, prioritize external validation on:
   - a second architecture
   - a second optimizer
   - a closely related task

For this repo, the working interpretation is:

- Phase 1 is already handled in `../dashifine`
- the main job here is to translate `LeechTransformer/` into an appropriate second-architecture test against the DASHI baseline

## Repo-Relevant State

Current local scripts that support this line of work:

- `26_grok_critical_scan.py`
  - Tracks `t95` across near-critical weight decays and cross-prime checks.
- `26_grok_sweep_adaptive.py`
  - Narrows the grokking threshold region.
- `26_grok_sweep_adaptive_spv2.py`
  - Parameterized scan for longer reproducible sweeps.

These scripts already support the broad "time-to-grok" benchmarking story, but the current repo did not yet contain the flagship note or documentation for the stronger `t50`/shared-onset logistic law.

## Roadmap Position

The current intended phase ordering is:

1. Take the grokking law from `../dashifine` as the benchmark baseline.
2. Validate it externally from this repo, starting with Leech architecture ablations before changing optimizer or task.
3. Compare compression-first learning against architectural priors.
4. Study geometry and mechanism.
5. Hand broader formalism questions to `../dashi_agda`.

Detailed staging lives in `ROADMAP.md`.

## Relationship To Older DASHI Context

The earlier "Repo Comparison: Sovereign-Lila-E8 vs Dashi" thread remains historically relevant, especially for longer-term DASHI formalism and geometry goals. However, it is no longer the active canonical thread for immediate repo planning. The current active planning surface should prioritize the grokking-law documentation and its follow-on measurements before returning to the broader DASHI formalism stack.

## Immediate Follow-On Work

1. Keep `GROKKING_TIME_RESCALING_NOTE.md` aligned with the accepted Phase 1 baseline coming from `../dashifine`.
2. Treat the current translated Leech result as partial transfer evidence:
   - logistic family still plausible
   - shared onset shifted strongly
   - timing screen much noisier than baseline
3. Run the architecture-ablation ladder first:
   - `lambda_geo = 0`
   - `lambda_geo = 1e-3`
   - `lambda_geo = 1e-2` control
4. Only after that, move to second-optimizer and related-task validation.
5. Define a shared validation/report surface for:
   - `t50`
   - `t95`
   - onset fraction
   - fit error
   - trajectory-shape notes
6. Treat external validation as the highest-value next scientific test in this repo.

## Notes

- The thread was fetched online by UUID and persisted into `~/chat_archive.sqlite`, then resolved locally from the DB.
- Resolved source:
  - `source = db`
  - `decision_reason = db_match_found`
  - `match_type = online_thread_id_exact`
- Thread title in the archive: `Grokking Valuation Resolution`
