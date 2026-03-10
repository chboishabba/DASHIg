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

## Current Phase 2 Leech Readout

The current translated Leech comparison does not reproduce the accepted baseline cleanly, but the latest canonical thread sharpens how to read that failure.

What currently appears true:

- the translated Leech model still groks the modular task
- a logistic rise still fits better than Gompertz in the current Leech runs
- the normalized onset shifts substantially earlier than the accepted baseline:
  - baseline `shared_c ≈ 0.8055`
  - current Leech `shared_c ≈ 0.3254`
- the clean inverse-weight-decay timing screen becomes much noisier than in the baseline

What the first architecture-ablation prelim adds:

- removing the geometric penalty (`lambda_geo = 0`) on the representative `wd = 0.22, 0.30` slice still yields clean grokking
- but it does not recover the baseline onset law:
  - prelim `shared_c ≈ 0.3893`
  - fixed logistic MSE `≈ 0.0292`
  - fitted logistic MSE `≈ 0.0238`
- so resonance penalty alone does not appear to be the whole explanation
- however, the apparent perfect `t95` fit in that prelim is not scientifically meaningful because it uses only two weight-decay points

So the current reading is not simply "Leech destroyed the law." A better statement is:

- the baseline law does not transfer cleanly under the current translated Leech setup
- some growth-family evidence may remain
- the observation channel / basis may be shifting the apparent onset and curve shape

This keeps architecture ablation as the immediate next step before broader optimizer/task variation, but narrows the hypothesis:

- the translated basis / architecture still looks like a live source of the mismatch
- the next useful discriminator is now `lambda_geo = 1e-3`, because the neutral plain-transformer prelim is already in hand

What the neutral standard-baseline prelim adds:

- the plain transformer also groks cleanly on the representative `wd = 0.22, 0.30` slice
- it does not match the DASHI baseline law either:
  - prelim `shared_c ≈ 0.3397`
  - fixed logistic MSE `≈ 0.0228`
  - fitted logistic MSE `≈ 0.0223`
- on this small slice, it is in the same broad timing regime as the Leech prelim rather than being obviously dominated by it

What the first FFT spike test adds:

- using the current local `test_loss` trajectories, both Leech and the plain baseline show low-frequency components, but they are broadly similar rather than clearly architecture-specific
- Leech prelim low-frequency dominant periods land around `696` to `910` epochs
- plain-baseline low-frequency dominant periods land around `724` to `820` epochs
- so, on the data currently in hand, FFT does not yet support a distinctive Leech-only spike frequency story

This is not a strong negative result against the external E8/Leech claim, because:

- these are local modular benchmark runs, not the long external lattice-language runs
- logging is every `20` epochs rather than every training step
- and the representative-band prelims are much shorter than the `140k+` step regime discussed externally

What the first derivative-shape test adds:

- the normalized rise profiles for both Leech and the plain baseline are much more interpretable than the FFT output
- both look broadly bell-shaped after `t50` normalization, which is consistent with a logistic-like underlying transition plus non-periodic instability spikes
- on the current tiny slice, Leech's mean derivative peak lands slightly earlier than the plain baseline:
  - Leech mean `peak_x ≈ 1.07`
  - plain baseline mean `peak_x ≈ 1.27`
- but the sample is only two runs per model, and the within-model shape correlations are only moderate (`~0.76`), so this is not yet strong evidence for a universal shared derivative family

So, from the data currently in hand, there is not yet a strong basis for an outreach claim like "Leech is almost 10x better than DASHI and clearly better than a standard baseline." The current evidence is narrower:

- both translated Leech and a neutral standard baseline generalize much earlier than the accepted DASHI baseline on this representative slice
- neither currently reproduces the DASHI timing law
- and the neutral standard baseline is competitive with the Leech prelim on the same slice
- the first FFT spike test does not isolate a unique Leech resonance signature on the local modular runs
- the derivative-shape test is more promising than FFT and is consistent with the idea that the invariant may be the growth curve rather than the spikes

## Immediate Next Work

1. Keep the accepted Phase 1 baseline from `../dashifine` fixed.
2. Finish the representative-band Leech prelims:
   - `lambda_geo = 1e-3`
   - compare against the existing `lambda_geo = 0` prelim and `lambda_geo = 1e-2` control
3. Compare the existing plain-baseline prelim against the `lambda_geo = 0` Leech prelim and the `lambda_geo = 1e-2` control.
4. Run the `lambda_geo = 1e-3` Leech prelim on the same representative band.
5. Decide from those three prelims whether the full overnight ladder is worth running.
6. Only after that, run broader external validation on:
   - second architecture
   - second optimizer
   - closely related task
7. If external raw step-level E8/Leech logs become available, rerun the FFT spike test on those logs directly rather than extrapolating from the local modular harness.
8. Extend the derivative-shape analysis to compare Leech, plain baseline, and the accepted DASHI baseline on a larger scan before making universality claims.

## Boundaries

- This note documents intended interpretation and next analysis targets.
- It does not claim that the `t50`/shared-onset pipeline is implemented or owned in this repo.
- Mechanistic explanations such as metastable escape should be treated as follow-on interpretation, not as the primary result.
- Broader DASHI formalism work remains important, but it is owned in `../dashi_agda`, outside the immediate scope of this note.
