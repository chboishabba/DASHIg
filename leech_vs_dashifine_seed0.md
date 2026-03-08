# Phase 2 Comparison

| label | architecture | optimizer | shared_c | fixed_mse | fitted_mse | t95~1/wd r2 | align t50 | align t95 | final test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dashifine_baseline | dashi_baseline | adamw | 0.8054802151744889 | 0.0003599398250397975 | 0.00035069122389562224 | 0.9976373410738996 | 5.150011836672701e-05 | 0.00021471642178932983 |  |
| leech_mul_adamw_seed0 | translated_leech_modular_classifier | adamw | 0.32540211256280865 | 0.02824960653876869 | 0.02718519615158297 | 0.38656513035879725 | 0.005163533070011746 | 0.012144745799311778 | 1.0 |

Baseline source note: `../dashifine/GROKKING_TIME_RESCALING_NOTE.md`
