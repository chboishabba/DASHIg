# External Log Alignment Template

Use this template when an external run provides some mix of:

- stepwise loss / validation logs
- stable-rank or condition-number measurements
- checkpointed samples
- anecdotal observations about semantic phases

The goal is to put all of them onto one comparable surface before interpreting mechanism.

## Run Metadata

- source repo / issue / thread:
- model / architecture:
- dataset / domain:
- optimizer:
- sampling settings:
- hardware path:
- supported runtime: `yes/no`

## Timing Surface

Record all available quantities against the same step axis.

| step | train_loss | val_loss | val_acc | lr | wd | t50_norm_x | derivative_mass | derivative_peak_local | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Representation Surface

| step | layer | stable_rank | condition_number | notes |
| --- | --- | --- | --- | --- |

Questions:

- is stable rank still dropping, flat, or rebounding?
- do condition numbers rise everywhere or only in some layers?
- does a representation change lead, lag, or coincide with the transition metrics?

## Checkpointed Sample Audit

Use fixed prompts across checkpoints.

| step | prompt_id | runtime_path | sample_summary | anomaly_score | violence_or_horror_proxy | human_note |
| --- | --- | --- | --- | --- | --- | --- |

Questions:

- does any reported "bad mode" phase reproduce on CPU?
- does it survive a supported accelerator path?
- is there a clean boundary in step-space where the mode disappears?
- does that boundary align with derivative mass or representation change?

## Runtime Validation

Before treating strange outputs as model behavior:

- compare CPU vs GPU on the same checkpoint and prompt
- compare supported vs unsupported accelerator paths
- record dtype
- record sampling parameters such as `top_p`, `top_k`, and temperature
- verify all tensors and the model are on the intended device
- record whether the host session remains stable for the full run or hits a reset / black-screen failure

If the effect appears only on an unsupported runtime path, classify it as a runtime artifact until reproduced elsewhere.
If the lane crashes the host session after some runtime window, classify it as operationally unsafe even if the workload starts successfully.

## Provisional Interpretation

Choose one:

- growth-law evidence only
- growth-law plus representation-compression evidence
- possible bad-intermediate-mode evidence
- runtime artifact, not usable for mechanism claims

## Decision

- what is supported now:
- what remains anecdotal:
- next measurement to run:
