# DASHIg

This repository is the Phase 2 comparison and validation harness for grokking dynamics work.

Phase ownership is split intentionally:

- Phase 1 baseline (`t50`/shared-onset/logistic law): `../dashifine`
- Phase 2 external validation and architecture comparison: this repo
- Broader formalism track: `../dashi_agda`

The current question here is not "does grokking exist?" but:

- how stable is the normalized growth law across architectures
- whether geometry changes curve family or primarily timing/scale

## Current Status

Current representative-band results (`wd = 0.22, 0.30`) are in-repo:

- translated Leech prelim: `leech_arch_ablation_prelim/`
- plain transformer prelim: `plain_baseline_prelim/`
- derivative comparison table: `derivative_comparison_prelim.csv`

Short read:

- both Leech and plain baseline grok much earlier than the accepted DASHI baseline on the tiny prelim slice
- Leech is not yet clearly dominant vs the plain baseline on that slice
- FFT on local modular runs is non-distinctive
- derivative-shape analysis is more informative and currently suggests a shared growth-family signal with timing differences
- external discussion in `SPUTNIKAI/sovereign-lila-e8#3` now adds two follow-on hypotheses to test locally:
  - continued pretraining may further compress stable rank rather than expand it
  - stronger geometric priors may suppress bad intermediate modes, but this remains a hypothesis until raw logs and sample audits are aligned with the existing timing surface
- the same external thread also strengthens one negative constraint:
  - unsupported GPU-path garbage output should be treated as a hardware/runtime problem, not as model-behavior evidence, when CPU inference is normal
  - the extracted GPU wrapper may start correctly but still trips the known KFD-reset / black-screen failure class after roughly `15` to `30` minutes on this host, so it is not a safe routine compute path yet

See:

- `ROADMAP.md`
- `TODO.md`
- `GROKKING_TIME_RESCALING_NOTE.md`
- `COMPACTIFIED_CONTEXT.md`

## Main Scripts

Benchmark runners:

- `27_leech_grok_critical_scan.py`
- `30_plain_grok_critical_scan.py`
- `28_leech_arch_ablation.py`
- `29_leech_arch_ablation_prelim.py`
- `31_plain_baseline_prelim.py`

Analysis tools:

- `27_leech_trajectory_analysis.py`
- `27_compare_to_dashifine_baseline.py`
- `32_fft_spike_analysis.py`
- `33_logistic_derivative_analysis.py`
- `34_derivative_comparison_table.py`
- `35_lila_log_to_csv.py`
- `36_lila_training_dynamics.py`
- `37_lila_delta_cone_analysis.py`
- `38_lila_phase2_pipeline.sh`

Model implementations:

- `phase2_validation/leech_modular_benchmark.py`
- `phase2_validation/plain_modular_benchmark.py`

## Typical Workflow

1. Run benchmark scans/prelims.
2. Run trajectory analysis.
3. Compare against the accepted `../dashifine` baseline.
4. Run derivative/FFT diagnostics as needed.
5. If external LILA logs are available, adapt them onto the same Phase 2 surface.
6. Update docs/TODO/changelog before changing direction.

The current immediate next step is:

- run `lambda_geo = 1e-3` on the same representative band
- then compare `lambda_geo = 0`, `1e-3`, `1e-2`, and plain baseline before deciding whether to run the full overnight ladder

## Example Commands

Leech representative prelim:

```bash
python 29_leech_arch_ablation_prelim.py
```

Plain baseline representative prelim:

```bash
python 31_plain_baseline_prelim.py
```

Derivative analysis for one run output:

```bash
python 33_logistic_derivative_analysis.py \
  --summary leech_arch_ablation_prelim/mul_adamw_lambda_0/scan.csv \
  --trajectories leech_arch_ablation_prelim/mul_adamw_lambda_0/scan_trajectories.csv \
  --label leech_lambda_0 \
  --out-prefix leech_arch_ablation_prelim/mul_adamw_lambda_0/derivative_shape
```

Combine derivative summaries:

```bash
python 34_derivative_comparison_table.py \
  --summaries \
    leech_arch_ablation_prelim/mul_adamw_lambda_0/derivative_shape_summary.csv \
    plain_baseline_prelim/derivative_shape_summary.csv \
  --out-prefix derivative_comparison_prelim
```

External LILA log adaptation:

```bash
./38_lila_phase2_pipeline.sh LeechTransformer/train_logs/340K.md
```

## Notes

- The root worktree may contain generated artifacts from analysis runs.
- Treat representative-band (`wd = 0.22, 0.30`) outcomes as directional, not final.
- Larger-seed claims should be made only after broader runs are complete.
- External anecdotes about semantic stability or safety should be treated as prompts for measurement, not as settled architecture claims.
- External LILA/Leech logs should be adapted onto the same timing surface before
  they are used in mechanism arguments.
- Unsupported inference-path failures should be excluded from semantic claims unless they reproduce on CPU or on a supported accelerator stack.
- A GPU lane that starts but crashes the host session is still not a usable experiment surface for unattended runs.

- `37_lila_delta_cone_analysis.py` can also emit bad-mode suppression summaries when a `basin` column or `coherence_score`/`bad_mode_score` columns are present.
