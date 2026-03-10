# ML Roadmap

## Summary

This repo is pursuing a comparative machine-learning program rather than a single isolated grokking result. The guiding question is how useful structure and symmetry arise in learning systems, and which route produces the clearest, fastest, and most transferable behavior:

- discovered via compression-driven dynamics
- injected as an architectural prior
- analyzed through observer/entropy geometry

The current active result is the grokking time-rescaling law, but Phase 1 ownership now lives upstream in `../dashifine`. This repo starts from Phase 2: external validation and comparison, with `LeechTransformer/` as the main second-architecture candidate and `../dashi_agda` carrying the separate broader formalism track.

## Program Tracks

### 1. Compression-First Learning

This is the main DASHI-facing track. It uses grokking experiments and related learners to test whether stable structure emerges from learning dynamics rather than from hard-coded symmetry.

Primary baseline ownership:

- `../dashifine` for the Phase 1 law and its local reproduction artifacts
- `../dashi_agda` for the broader formal framing beyond the benchmark layer

Current concrete assets:

- `26_grok_critical_scan.py`
- `26_grok_sweep_adaptive.py`
- `26_grok_sweep_adaptive_spv2.py`
- `GROKKING_TIME_RESCALING_NOTE.md`

Core benchmark:

- time-to-generalization
- trajectory shape under time normalization
- representation/geometry diagnostics after the law is stable

### 2. Architectural-Prior Learning

This track uses explicit geometric priors as a comparison point for compression-first discovery. The current concrete instance is `LeechTransformer/`, which builds Leech-lattice structure directly into the model.

Question:

- what is gained or lost when symmetry is injected architecturally instead of discovered dynamically?

### 3. Observer / Entropy Geometry

This track uses `cognitive-observer-simulation/` to study entropy scaling, phase transitions, and representational geometry in an analytical simulation setting.

Question:

- can entropy/phase-transition analyses help interpret or compare the structures found in the learning tracks?

## Phase Order

### Phase 1. Lock the Grokking Law

Goal:

- treat the current flagship empirical law as the accepted upstream baseline for this repo.

Primary owner:

- `../dashifine`

Required inputs into this repo:

- baseline `t50` extraction
- shared-onset fit summary
- logistic post-onset fit summary
- comparison of `t50` vs `t95`

### Phase 2. External Validation

Goal:

- test whether the law survives outside the original setup and whether the LeechTransformer mechanism can be translated into a defensible second-architecture comparison against the DASHI baseline.

Highest-value checks:

- second architecture via `LeechTransformer/`
- neutral industry-standard baseline
- second optimizer
- closely related task

This is the main job of this repo. External validation matters more than expanding interpretations before the law is tested elsewhere.

Current Phase 2 reading:

- the translated Leech modular classifier clearly solves the task
- logistic growth still fits better than Gompertz in the current Leech runs
- the shared normalized onset moves substantially earlier than the baseline (`~0.325` vs `~0.8055`)
- the inverse-weight-decay timing screen is much noisier than in the baseline
- the first `lambda_geo = 0` prelim on `wd = 0.22, 0.30` does not restore the baseline law:
  - `shared_c ≈ 0.389`
  - fixed logistic MSE `≈ 0.0292`
  - fitted logistic MSE `≈ 0.0238`
  - both points still grok cleanly
- the neutral plain-transformer prelim on the same slice also groks cleanly:
  - `shared_c ≈ 0.340`
  - fixed logistic MSE `≈ 0.0228`
  - fitted logistic MSE `≈ 0.0223`
  - it is in the same broad timing regime as the Leech prelim, not an obviously dominated baseline

Important boundary on that prelim:

- the apparent `t95 ~ 1 / wd` `R^2 = 1.0` is from only two points and should not be treated as evidence of a clean recovered screen

So the immediate next step is still not "run longer" and not yet "change optimizer/task". The `lambda_geo = 0` prelim suggests resonance penalty alone is not the whole story, and the neutral baseline prelim means there is not yet a strong case that Leech is uniquely faster than standard architectures on this slice. The next discriminating check is therefore:

- `lambda_geo = 1e-3` prelim on the same representative band

Sub-phase ordering inside Phase 2:

1. Architecture ablation first:
   - `lambda_geo = 0`
   - `lambda_geo = 1e-3`
   - `lambda_geo = 1e-2` control
2. Neutral standard baseline on the same modular benchmark surface:
   - plain small transformer or MLP
   - same task, split, optimizer family, and report surface
3. Use the three representative-band prelims (`lambda_geo = 0`, `lambda_geo = 1e-3`, plain baseline) to decide whether the full overnight ladder is justified before spending that runtime.
4. Second optimizer after at least one Leech or standard-baseline result looks scientifically comparable.
5. Closely related task after that.

### Phase 3. Cross-Paradigm Comparison

Goal:

- compare the upstream compression-first baseline against architectural priors using a shared reporting surface.

Primary comparator:

- `LeechTransformer/`

Required comparison step:

- translate the LeechTransformer mechanism into an apples-to-apples test surface against the DASHI baseline rather than comparing raw repositories at face value

Shared comparison axes:

- relative speedup against a neutral standard baseline
- time-to-generalization
- trajectory-law stability
- onset fraction / growth-factor stability
- representation geometry
- training stability / sensitivity

### Phase 4. Geometry and Mechanism

Goal:

- study why the law appears, without replacing the law itself with interpretation.

Work in this phase:

- geometry-after-grok analysis
- metastable-escape interpretation checks
- observer/entropy analyses as interpretive tools

Rule:

- mechanism remains secondary to empirical validation until Phase 2 is completed
- if Leech preserves growth-family evidence but shifts onset/shape, document that as partial transfer rather than forcing a binary success/failure reading

### Phase 5. Return to Broader DASHI Formalism

Goal:

- hand validated ML findings back to the broader DASHI formalism effort.

Primary owner:

- `../dashi_agda`

Deferred topics:

- formalism write-up
- prime-based learners beyond the current benchmark layer
- broader geometry/physics closure agenda

## Current Defaults

- flagship empirical claim: shared-onset logistic law under `t50` normalization
- immediate next scientific step in this repo: finish the `lambda_geo = 1e-3` representative-band Leech prelim, then compare it against the existing `lambda_geo = 0`, `lambda_geo = 1e-2`, and plain-baseline readouts before making strong external claims about relative speedups
- mechanism status: metastable delayed plateau is interpretation, not flagship claim
- roadmap audience: internal research planning for this repo and its embedded comparison projects
