# Dashifine Baseline Reference

This file records the accepted Phase 1 baseline manually so the local Phase 2
comparison code does not depend on runtime reads from `../dashifine`.

## Accepted Claim

After time rescaling by `t50`, onset occurs near a shared normalized location
`t0 ~= 0.81 * t50`, and the post-escape rise is well fit by one shared logistic
curve.

## Source Artifacts

- `../dashifine/grok_analysis_combined/grok_milestones.csv`
- `../dashifine/grok_analysis_combined/grok_onset_fit_screen.csv`
- `../dashifine/grok_analysis_combined/grok_rise_logistic_fitted_t0_fit.csv`
- `../dashifine/grok_analysis_combined/grok_rise_logistic_fixed_ct50_fit.csv`
- `../dashifine/GROKKING_TIME_RESCALING_NOTE.md`

## Accepted Numbers

- `t95 ~ 1 / weight_decay`:
  - `r2 ~= 0.9976373410738996`
- alignment under normalization:
  - `alignment_mse_norm_t50 ~= 5.150011836672701e-05`
  - `alignment_mse_norm_t95 ~= 0.0001384772978034187`
- fixed shared-onset logistic fit:
  - `k ~= 12.34619689833151`
  - `x0 ~= 0.19050749921914506`
  - `mse ~= 0.0003599398250397975`
  - `shared_c ~= 0.8054802151744889`
- fitted-onset logistic fit:
  - `k ~= 12.343957106398715`
  - `x0 ~= 0.18114050505829873`
  - `mse ~= 0.00035069122389562224`

## Expected Comparison Surface

- task
- architecture
- optimizer
- `t_fit`
- `t50`
- `t95`
- shared-onset coefficient `c`
- fixed and fitted logistic MSE
- `t95 ~ 1 / weight_decay` fit quality
- `t50` vs `t95` alignment quality
- final train/test accuracy
- short trajectory note
