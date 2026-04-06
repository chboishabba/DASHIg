# LILA Phase 2 Validation

This note is the `DASHIg` entrypoint for external LILA or Leech logs.

The job here is empirical:

- adapt raw LILA logs onto the repo's Phase 2 timing surface
- compare transition timing and derivative shape against existing baselines
- treat stable-rank / condition-number trends and bad-intermediate-mode stories
  as aligned measurement targets rather than as settled mechanism

## Files

- `35_lila_log_to_csv.py`
- `36_lila_training_dynamics.py`
- `37_lila_delta_cone_analysis.py`
- `38_lila_phase2_pipeline.sh`
- `LILA_PHASE2_VALIDATION.puml`
- `LILA_PHASE2_VALIDATION.svg`

## One-command path

```bash
./38_lila_phase2_pipeline.sh LeechTransformer/train_logs/340K.md
```

This writes:

- adapted CSV with `step`, `train_loss`, `val_loss`
- optional aligned representation columns when `SR_CSV` is provided
- training-dynamics plot
- delta-cone ranking CSV
- optional bad-mode suppression summary JSON and basin-transition CSV when basin labels or score columns are available

If you have stable-rank / condition-number logs from the notebook path, merge them on the same step axis:

```bash
SR_CSV=stable_rank_log.csv ./38_lila_phase2_pipeline.sh LeechTransformer/train_logs/340K.md
```

## Interpretation boundary

Use these artifacts to answer:

- where transition mass concentrates in step-space
- whether raw LILA timing is comparable to the existing Phase 2 surfaces
- whether aligned representation channels such as stable rank or condition
  number lead, lag, or coincide with those transitions
- whether coherent-bad occupancy decreases after coherence has formed

Do not treat the output as a proof that geometry enforces safety.
Treat it as a way to put external LILA evidence onto the same comparison plane
already used in this repo.
