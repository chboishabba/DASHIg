# Phase 2 Comparison

| label | architecture | optimizer | shared_c | fixed_mse | fitted_mse | t95~1/wd r2 | align t50 | align t95 | final test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dashifine_baseline | dashi_baseline | adamw | 0.8054802151744889 | 0.0003599398250397975 | 0.00035069122389562224 | 0.9976373410738996 | 5.150011836672701e-05 | 0.00021471642178932983 |  |
| plain_baseline_prelim | plain_modular_transformer | adamw | 0.3396750557842165 | 0.022822662284937983 | 0.02232811675001422 | 1.0 | 0.0022503715292274845 | 0.007787941840965455 | 1.0 |

Baseline source note: `../dashifine/GROKKING_TIME_RESCALING_NOTE.md`
