# TODO

## Context Sync

- [x] Pull online thread `6958ff8a-03c8-8321-b906-30e48e412a3a` into `~/chat_archive.sqlite` and resolve the canonical local thread.
- [x] Replace the stale repo-comparison context with the active theorem-assistance/grokking-law thread.
- [x] Add a repo-level `ROADMAP.md` to reconcile the active grokking thread with the broader ML program in this repo.
- [x] Record the ownership split: `../dashifine` for Phase 1 baseline, this repo for Phase 2 comparison/validation, and `../dashi_agda` for broader formalism.

## External Dependency: Phase 1 Baseline (`../dashifine`)

- [x] Create `GROKKING_TIME_RESCALING_NOTE.md` as the primary note for the current empirical law.
- [ ] Import or reference the accepted baseline artifacts from `../dashifine` so this repo has a stable comparison target.
- [ ] Record the exact baseline report surface expected from upstream: `t50`, `t95`, onset fraction, fit error, and trajectory-shape notes.

## Phase 2: External Validation and Translation

- [ ] Analyze `LeechTransformer/` and identify which parts of its mechanism can be translated into a second-architecture test against the DASHI baseline.
- [ ] Define the translation target explicitly: what constitutes an apples-to-apples comparison between LeechTransformer behavior and the DASHI baseline from `../dashifine`.
- [ ] Re-run the law on a second architecture, with `LeechTransformer/` as the default first comparator.
- [ ] Re-run the law with a second optimizer to test whether the normalized onset/logistic story is optimizer-stable.
- [ ] Re-run the law on a closely related task to test transfer beyond the original setup.
- [ ] Standardize a shared report template for all external-validation runs: `t50`, `t95`, onset fraction, fit error, and qualitative trajectory notes.

## Phase 3: Cross-Paradigm Comparison

- [ ] Compare the `../dashifine` compression-first baseline against the translated architectural-prior test in `LeechTransformer/`.
- [ ] Define a common comparison table covering time-to-generalization, trajectory-law stability, representation geometry, and training sensitivity.

## Phase 4: Geometry and Mechanism

- [ ] Evaluate the metastable-escape interpretation against the new collapse law rather than treating it as a standalone story.
- [ ] Run geometry-after-grok analyses only after at least one external-validation setting reproduces the law.
- [ ] Use `cognitive-observer-simulation/` as an interpretation/analysis branch rather than as the immediate benchmark driver.

## External Dependency: Broader DASHI Work (`../dashi_agda`)

- [ ] Hand validated ML findings back to `../dashi_agda` once the comparison work here is mature enough to inform the formal track.
