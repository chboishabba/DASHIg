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
- [x] Add a short prelim ablation runner for one-leg, representative-band checks before committing to the full overnight ladder.
- [x] Run the first translated Leech second-architecture sweep on modular multiplication with AdamW and compare it against the accepted `../dashifine` baseline.
- [x] Record the current Phase 2 interpretation: solvability transfers; logistic-family evidence remains; shared onset shifts strongly earlier; clean `t95 ~ 1 / wd` behavior degrades.
- [ ] Run the Leech architecture-ablation ladder first:
  - `lambda_geo = 0`
  - `lambda_geo = 1e-3`
  - `lambda_geo = 1e-2` control
- [x] Run the representative-band `lambda_geo = 0` prelim on `wd = 0.22, 0.30`.
- [x] Record the `lambda_geo = 0` prelim result: clean grokking persists, but `shared_c ≈ 0.389` still sits far from the DASHI baseline and does not restore the law.
- [ ] Run the representative-band `lambda_geo = 1e-3` prelim on `wd = 0.22, 0.30`.
- [ ] Decide from the ablation whether the current mismatch is driven mainly by resonance penalty strength, frozen geometry, or basis/channel effects.
- [x] Add a neutral industry-standard modular benchmark implementation on the same task/logging surface as the translated Leech runs.
- [x] Add a short prelim entrypoint for the neutral standard baseline so it can be screened on the representative weight-decay slice quickly.
- [x] Run the neutral standard-baseline prelim on `wd = 0.22, 0.30` and compare it against both the accepted `../dashifine` baseline and the Leech prelims before outreach or strong speedup claims.
- [x] Record the current standard-baseline prelim result: it is competitive with the `lambda_geo = 0` Leech prelim on the representative slice, so there is no strong standard-baseline dominance claim yet.
- [x] Run an FFT spike test on the current local Leech and plain-baseline `test_loss` trajectories.
- [x] Record the FFT result: both local models show broadly similar low-frequency components, so there is no distinctive Leech-only spike spacing signal on the current modular runs.
- [x] Run derivative-shape analysis on the current normalized growth curves to test whether Leech and the plain baseline share a common bell-shaped rise profile even when spike structure differs.
- [x] Run derivative-shape analysis on the current normalized growth curves for the Leech and plain representative-band prelims.
- [x] Record the derivative result: both models show bell-shaped normalized rise profiles, and Leech peaks slightly earlier on average on the current tiny slice, but the sample is too small for strong universality claims.
- [x] Extend derivative-shape analysis to emit pre-`t50` area, slope proxy, and a compact summary row suitable for a 4-way representative-band comparison table.
- [x] Add a compact derivative-comparison table tool for `lambda_geo = 0`, `1e-3`, `1e-2`, and plain-baseline summaries.
- [ ] If external raw step-level E8/Leech logs are obtained, rerun the FFT spike test on those logs directly.
- [ ] If external raw step-level E8/Leech logs are obtained, align them with stable-rank / condition-number measurements where available rather than treating loss spikes in isolation.
- [x] Add a thin external-LILA log adapter, timing plotter, and delta-cone scan so
  `LeechTransformer/train_logs/*.md` can be put onto the same Phase 2 surface
  without manual spreadsheet work.
- [ ] Extend derivative-shape analysis to the broader Leech/plain/DASHI comparison surface if the full ladder is run.
- [ ] Use the `lambda_geo = 0`, `lambda_geo = 1e-3`, `lambda_geo = 1e-2`, and plain-baseline representative-band results to decide whether the full overnight ladder is justified.
- [ ] Re-run the law with a second optimizer after at least one Leech ablation leg looks scientifically comparable.
- [ ] Re-run the law on a closely related task after the architecture question is cleaner.
- [x] Standardize a shared report template for all external-validation runs: `t50`, `t95`, onset fraction, fit error, and qualitative trajectory notes.
- [ ] Add an external-log comparison template that can join:
  - stepwise loss / validation
  - derivative-transition metrics
  - stable rank / condition number
  - checkpointed sample audits
- [ ] Record the transcribed external `345k` -> `400k` stable-rank / condition-number continuation as additional compression-under-domain-shift evidence and watch for plateau vs continued descent.
- [x] Add optional bad-mode suppression summaries to the external delta-cone scan: basin dwell fractions, bad/safe run lengths, and bad->safe transition counts.
- [ ] If the "horror phase" anecdote is pursued, define a measurable proxy for bad intermediate modes before interpreting it as geometry-induced safety.
- [ ] Define a simple checkpoint-audit surface for the "bad intermediate mode" story:
  - fixed prompts
  - toxicity / violence / anomaly proxy
  - human notes
  - alignment to training step and derivative mass
- [ ] Record the continued-pretraining stable-rank result from `SPUTNIKAI/sovereign-lila-e8#3` as hypothesis-generating evidence for compression-under-domain-shift, then test whether the same pattern appears on any locally reproducible surface.
- [ ] Add a runtime-validation checklist for external inference anecdotes:
  - CPU vs GPU
  - supported vs unsupported accelerator path
  - dtype / sampling settings
  - same prompt, same checkpoint
- [ ] Record the extracted GPU-wrapper result as partial bring-up only:
  - workload starts on GPU
  - host hits the known KFD-reset / black-screen failure after about `15` to `30` minutes
  - do not treat this lane as safe for unattended ablation runs
- [ ] If the gfx803 GPU lane is revisited, characterize the stability envelope explicitly:
  - shortest reproducible time-to-reset
  - prompt length / batch / token sensitivity
  - whether `HIP_LAUNCH_BLOCKING=1` or solver restrictions change failure timing
  - whether emergency-shell recovery is possible on any variant

## Phase 3: Cross-Paradigm Comparison

- [ ] Compare the `../dashifine` compression-first baseline against the translated architectural-prior test in `LeechTransformer/`, including partial-transfer outcomes where growth family survives but onset/shape shifts.
- [ ] Define a common comparison table covering time-to-generalization, trajectory-law stability, representation geometry, and training sensitivity.

## Phase 4: Geometry and Mechanism

- [ ] Evaluate the metastable-escape interpretation against the new collapse law rather than treating it as a standalone story.
- [ ] Evaluate the "shadow structure" / inverse-mode interpretation against checkpointed outputs and representation diagnostics rather than prose analogy alone.
- [ ] Evaluate whether continued compression under domain shift is better described as global crystallization, continued specialization, or a measurement artifact.
- [ ] Separate semantic corruption claims from host-stability failures when summarizing external accelerator evidence.
- [ ] Run geometry-after-grok analyses only after at least one external-validation setting reproduces the law.
- [ ] Use `cognitive-observer-simulation/` as an interpretation/analysis branch rather than as the immediate benchmark driver.

## External Dependency: Broader DASHI Work (`../dashi_agda`)

- [ ] Hand validated ML findings back to `../dashi_agda` once the comparison work here is mature enough to inform the formal track.
