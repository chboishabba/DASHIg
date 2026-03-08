# Compactified Context

- Conversation: `Theorem Assistance` (ID: `6958ff8a-03c8-8321-b906-30e48e412a3a`).
- Canonical thread ID: `2aec04871d54bda5059cd98155cd7512f13ab503`.
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
- Priority order here: translate LeechTransformer into a valid second-architecture test -> run external validation -> compare paradigms -> geometry/mechanism.
- Planning change: broader DASHI formalism/visualization work remains relevant background, but it is owned in `../dashi_agda`, not here.
