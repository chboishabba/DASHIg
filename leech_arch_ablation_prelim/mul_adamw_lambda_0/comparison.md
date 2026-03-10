# Phase 2 Comparison

| label | architecture | optimizer | shared_c | fixed_mse | fitted_mse | t95~1/wd r2 | align t50 | align t95 | final test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dashifine_baseline | dashi_baseline | adamw | 0.8054802151744889 | 0.0003599398250397975 | 0.00035069122389562224 | 0.9976373410738996 | 5.150011836672701e-05 | 0.00021471642178932983 |  |
| mul_adamw_lambda_0 | translated_leech_modular_classifier | adamw | 0.389328342998692 | 0.029216056234243486 | 0.02384326903642144 | 1.0 | 0.0029104408929377665 | 0.008199408851053433 | 1.0 |

Baseline source note: `../dashifine/GROKKING_TIME_RESCALING_NOTE.md`
