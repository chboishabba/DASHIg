# Grokking Time-Rescaling Note

## Status

This note is the current flagship documentation artifact for the grokking experiments tracked in this repo. It records the strongest empirical law surfaced in the latest canonical ChatGPT thread and separates established claims from analysis that still needs to be reproduced locally.

Within the repo-level ML roadmap, this note records the Phase 1 benchmark artifact, but Phase 1 ownership now lives in `../dashifine`. In this repo, the note exists to define the accepted baseline for downstream comparison work across compression-first learning, architectural priors, and observer/entropy analysis.

## Current Empirical Law

For weight decay `lambda` in the near-critical regime, the test-accuracy trajectory is well described by a shared curve after time rescaling by `t50`:

`A_test(t; lambda) ≈ F(t / t50(lambda))`

The current best summary is a three-part structure:

1. Time-rescaled trajectory collapse.
2. Shared normalized onset:
   - `t0 ≈ 0.8055 * t50`
3. Shared post-onset logistic rise.

This is a stronger statement than "late curves look similar" because it proposes both a natural clock (`t50`) and a nearly invariant onset location in that clock.

## Why This Matters

- It suggests weight decay primarily changes the timescale of generalization rather than the trajectory family.
- It turns late generalization from a qualitative observation into a compact law that can be falsified or extended.
- It provides a cleaner target for mechanism work, including metastable-escape interpretations.

## What Is Already Supported In This Repo

The repo already contains grokking sweep scripts that support the broader timing story:

- `26_grok_critical_scan.py`
  - scans near-critical weight decay and records `t95`
- `26_grok_sweep_adaptive.py`
  - adaptively narrows the grokking threshold region
- `26_grok_sweep_adaptive_spv2.py`
  - parameterized sweep runner for reproducible scans

These scripts support the timing/comparison workflow, but they do not yet document or export the full `t50`-based collapse law in this repo snapshot.

## Upstream Baseline Dependency

The latest fetched thread references fit outputs that are not currently present in this repo:

- `grok_rise_logistic_fixed_ct50_fit.csv`
  - shared onset coefficient `c ≈ 0.8055`
  - `mse ≈ 0.000360`
- `grok_rise_logistic_fitted_t0_fit.csv`
  - `mse ≈ 0.000351`

Until those artifacts are imported or referenced here, this note should treat them as the upstream baseline coming from `../dashifine`, not as locally owned deliverables in this repo.

## Immediate Next Work

1. Import or reference the accepted Phase 1 baseline from `../dashifine`.
2. Translate `LeechTransformer/` into an appropriate second-architecture test against that baseline.
3. Run external validation on:
   - second architecture
   - second optimizer
   - closely related task

## Boundaries

- This note documents intended interpretation and next analysis targets.
- It does not claim that the `t50`/shared-onset pipeline is implemented or owned in this repo.
- Mechanistic explanations such as metastable escape should be treated as follow-on interpretation, not as the primary result.
- Broader DASHI formalism work remains important, but it is owned in `../dashi_agda`, outside the immediate scope of this note.
