# Phase 2 Validation

This directory holds the local Phase 2 comparison contract for the DASHI
grokking-law work.

## Purpose

This repo does not own the accepted Phase 1 baseline. The accepted baseline
lives upstream in `../dashifine`. This directory records the manually curated
reference numbers and the comparison surface used by the local Phase 2 harness.

The local job is:

- translate the Leech architectural prior into a modular-arithmetic classifier
- run the same grokking-style trajectory analysis surface
- compare the translated Leech result against the accepted DASHI baseline

## Files

- `dashifine_shared_onset_logistic_baseline.json`
  - manual baseline summary copied from accepted upstream artifacts
- `dashifine_shared_onset_logistic_baseline.md`
  - human-readable baseline note

## Shared Report Surface

Every Phase 2 validation leg should report:

- task
- architecture
- optimizer
- train/test split
- `t_fit`
- `t50`
- `t95`
- shared-onset coefficient `c`
- fixed shared-onset logistic MSE
- fitted-onset logistic MSE
- `t95 ~ 1 / weight_decay` fit quality
- alignment error under `t50` normalization
- alignment error under `t95` normalization
- final train/test accuracy
- short qualitative note on whether the trajectory family looks preserved

## Current Defaults

- second architecture: translated Leech modular classifier
- second optimizer: SGD with momentum
- related task: modular addition
