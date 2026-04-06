# Compactified Context

- Conversation: `Grokking Valuation Resolution` (ID: `6958b536-7e18-8320-bce9-421436b4ccf2`).
- Canonical thread ID: `a450d4cb4d0be34146aab4df6898149e2910b472`.
- Storage: fetched online by UUID on 2026-03-08, persisted to `~/chat_archive.sqlite`, then resolved locally from DB.
- Repo-level ML framing now lives in `ROADMAP.md`.
- Core result: near-critical grokking trajectories collapse under time rescaling by `t50`.
- Strongest invariant: shared onset `t0 ≈ 0.8055 * t50`.
- Shape claim: after onset shift, the rise is well fit by a shared logistic curve.
- Fit quality called out in-thread:
  - fixed onset fit MSE `≈ 0.000360`
  - per-run onset fit MSE `≈ 0.000351`
- Current flagship task: write/maintain `GROKKING_TIME_RESCALING_NOTE.md` around this empirical law.
- Broader ML program: compare compression-first discovery, architectural priors, and observer/entropy analysis.
- Ownership split: `../dashifine` owns the Phase 1 baseline, this repo owns Phase 2 validation/comparison, and `../dashi_agda` owns broader formalism.
- Existing repo support: `26_grok_critical_scan.py`, `26_grok_sweep_adaptive.py`, `26_grok_sweep_adaptive_spv2.py`.
- Important distinction: this repo should consume the accepted Phase 1 law from `../dashifine`, not re-own it.
- Current Leech Phase 2 result: logistic still fits better than Gompertz, but the shared onset shifts to `c ≈ 0.3254` and the clean `t95 ~ 1 / wd` screen becomes noisy.
- `lambda_geo = 0` prelim on `wd = 0.22, 0.30` still groks cleanly but does not restore the DASHI onset law:
  - shared onset `c ≈ 0.3893`
  - fixed logistic MSE `≈ 0.0292`
  - fitted logistic MSE `≈ 0.0238`
- Neutral plain-transformer prelim on the same `wd = 0.22, 0.30` slice also groks cleanly:
  - shared onset `c ≈ 0.3397`
  - fixed logistic MSE `≈ 0.0228`
  - fitted logistic MSE `≈ 0.0223`
- Current comparison: on this tiny slice, Leech is not obviously dominating a neutral standard baseline; the plain baseline is in the same runtime/timing regime and even fits the logistic family slightly more cleanly.
- FFT check on current local `test_loss` trajectories does not show a unique long-period Leech resonance:
  - Leech low-frequency dominant periods are about `696` to `910` epochs
  - plain baseline low-frequency dominant periods are about `724` to `820` epochs
  - both are weak/moderate and broadly similar
- Derivative-shape check on normalized growth curves is more informative:
  - both Leech and plain baseline show bell-shaped rise profiles after `t50` normalization
  - Leech mean derivative peak is slightly earlier on this tiny slice (`~1.07`) than the plain baseline (`~1.27`)
  - shape correlations within each two-run set are moderate (`~0.76`), so the evidence is suggestive rather than decisive
- Current takeaway: removing the geometric penalty alone did not recover baseline-like timing; basis/architecture effects still look live.
- Working interpretation: the translated Leech model may preserve part of the growth law while changing the observed onset/shape through basis or channel effects.
- Priority order here: finish the `lambda_geo = 1e-3` Leech prelim -> compare all representative-band readouts, including FFT, side by side -> decide whether the full ladder is worth the overnight cost -> then optimizer/task validation.
- Planning change: broader DASHI formalism/visualization work remains relevant background, but it is owned in `../dashi_agda`, not here.
- External discussion also now includes GitHub issue `SPUTNIKAI/sovereign-lila-e8#3` (`imply time to grokking`, source: GitHub issue/comments, synced 2026-03-21).
- Main additions from that issue:
  - external Leech/Lila discussion now includes a "horror phase" anecdote for E8 but not Leech; locally this should be treated as a hypothesis about bad intermediate modes / shadow structure, not as evidence of inherent safety
  - external continued-pretraining logs now extend through `400k` and continue to report decreasing stable rank and rising condition numbers across sampled layers, suggesting further compression/specialization rather than dimensional expansion
  - transcribed `345k` -> `400k` continuation preserves the same ordering by layer: block `0` stays most compressed, block `11` least compressed among the three sampled blocks, while all three continue the same SR-down / CN-up trend
  - the strongest compatible local framing remains: architecture may constrain the search and change onset, width, asymmetry, or observation channel, while the normalized growth-family picture may still be the main invariant
  - immediate follow-on value from that issue is empirical, not rhetorical: align external raw logs, stable-rank trajectories, and sample-quality transitions with the existing `t50`/derivative analysis surface before making stronger mechanism or safety claims
  - the transcribed CPU-normal / GPU-garbage inference result should currently be treated as a hardware/runtime compatibility issue, not as evidence about model semantics
  - subsequent local use of the extracted GPU wrapper reportedly reached actual GPU execution but still triggered the known KFD-reset failure class after about `15` to `30` minutes: black screen, unrecoverable session, no emergency terminal
- New Phase 2 bridge lane for that issue:
  `35_lila_log_to_csv.py`,
  `36_lila_training_dynamics.py`,
  `37_lila_delta_cone_analysis.py`,
  and `38_lila_phase2_pipeline.sh`
  adapt raw `LeechTransformer/train_logs/*.md` logs onto the same timing surface
  already used for the modular comparison work.
