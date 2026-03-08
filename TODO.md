# TODO

## Context Sync

- [x] Pull online thread `6958ff8a-03c8-8321-b906-30e48e412a3a` into `~/chat_archive.sqlite` and resolve the canonical local thread.
- [x] Replace the stale repo-comparison context with the active theorem-assistance/grokking-law thread.
- [x] Add a repo-level `ROADMAP.md` to reconcile the active grokking thread with the broader ML program in this repo.
- [x] Record the ownership split: `../dashifine` for Phase 1 baseline, this repo for Phase 2 comparison/validation, and `../dashi_agda` for broader formalism.
- [x] Pull online thread `6958b536-7e18-8320-bce9-421436b4ccf2` into `~/chat_archive.sqlite` and update the canonical local context to `Grokking Valuation Resolution`.

## External Dependency: Phase 1 Baseline (`../dashifine`)

- [x] Create `GROKKING_TIME_RESCALING_NOTE.md` as the primary note for the current empirical law.
- [x] Import or reference the accepted baseline artifacts from `../dashifine` so this repo has a stable comparison target.
- [x] Record the exact baseline report surface expected from upstream: `t50`, `t95`, onset fraction, fit error, and trajectory-shape notes.

## Phase 2: External Validation and Translation

- [x] Analyze `LeechTransformer/` and identify which parts of its mechanism can be translated into a second-architecture test against the DASHI baseline.
- [x] Define the translation target explicitly: what constitutes an apples-to-apples comparison between LeechTransformer behavior and the DASHI baseline from `../dashifine`.
- [x] Add the translated Leech modular benchmark harness, local trajectory analysis, and baseline comparison scripts.
- [x] Add an ablation orchestration runner so the `lambda_geo` ladder can be executed and summarized as one Phase 2 job.
- [x] Run the first translated Leech second-architecture sweep on modular multiplication with AdamW and compare it against the accepted `../dashifine` baseline.
- [x] Record the current Phase 2 interpretation: solvability transfers; logistic-family evidence remains; shared onset shifts strongly earlier; clean `t95 ~ 1 / wd` behavior degrades.
- [ ] Run the Leech architecture-ablation ladder first:
  - `lambda_geo = 0`
  - `lambda_geo = 1e-3`
  - `lambda_geo = 1e-2` control
- [ ] Decide from the ablation whether the current mismatch is driven mainly by resonance penalty strength, frozen geometry, or basis/channel effects.
- [ ] Re-run the law with a second optimizer after at least one Leech ablation leg looks scientifically comparable.
- [ ] Re-run the law on a closely related task after the architecture question is cleaner.
- [x] Standardize a shared report template for all external-validation runs: `t50`, `t95`, onset fraction, fit error, and qualitative trajectory notes.

## Phase 3: Cross-Paradigm Comparison

- [ ] Compare the `../dashifine` compression-first baseline against the translated architectural-prior test in `LeechTransformer/`, including partial-transfer outcomes where growth family survives but onset/shape shifts.
- [ ] Define a common comparison table covering time-to-generalization, trajectory-law stability, representation geometry, and training sensitivity.

## Phase 4: Geometry and Mechanism

- [ ] Evaluate the metastable-escape interpretation against the new collapse law rather than treating it as a standalone story.
- [ ] Run geometry-after-grok analyses only after at least one external-validation setting reproduces the law.
- [ ] Use `cognitive-observer-simulation/` as an interpretation/analysis branch rather than as the immediate benchmark driver.

## External Dependency: Broader DASHI Work (`../dashi_agda`)

- [ ] Hand validated ML findings back to `../dashi_agda` once the comparison work here is mature enough to inform the formal track.
