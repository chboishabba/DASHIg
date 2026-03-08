# Changelog

## 2026-03-08

- Updated the canonical repo context from the older DASHI comparison thread to the fetched `Theorem Assistance` thread (`6958ff8a-03c8-8321-b906-30e48e412a3a` / `2aec04871d54bda5059cd98155cd7512f13ab503`).
- Added `GROKKING_TIME_RESCALING_NOTE.md` to document the current flagship empirical law: `t50` time-rescaling, shared onset `t0 ≈ 0.8055 * t50`, and shared post-onset logistic growth.
- Retargeted `TODO.md` toward local reproduction of the `t50`-based fits and away from treating the older DASHI formalism thread as the immediate active plan.
- Recorded the distinction between existing repo-local grokking scripts and the fit artifacts that are still external to this repo snapshot.
- Added `ROADMAP.md` to define the whole-program ML roadmap across compression-first learning, architectural priors, and observer/entropy analysis.
- Reframed the planning docs so the grokking-law note is Phase 1 of the broader ML program, with external validation as the next highest-value scientific step after local reproduction.
- Clarified ownership split: `../dashifine` owns the Phase 1 grokking-law baseline, this repo owns Phase 2 validation/comparison, and `../dashi_agda` owns the broader formalism track.
- Refocused this repo’s primary job on translating `LeechTransformer/` into a defensible second-architecture comparison against the DASHI baseline rather than re-owning baseline reproduction.
