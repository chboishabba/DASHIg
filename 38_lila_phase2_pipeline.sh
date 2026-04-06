#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./38_lila_phase2_pipeline.sh path/to/log.md [out_dir]
# Optional env:
#   SR_CSV=path/to/stable_rank_log.csv
#   COMPARE_CSV=path/to/baseline.csv

LOG_PATH=${1:?raw log path required}
OUT_DIR=${2:-lila_phase2_validation}
MODEL_NAME=${MODEL_NAME:-lila_external}
COMPARE_CSV=${COMPARE_CSV:-}
COMPARE_LABEL=${COMPARE_LABEL:-compare}
SR_CSV=${SR_CSV:-}

mkdir -p "$OUT_DIR"

CSV_PATH="$OUT_DIR/${MODEL_NAME}.csv"
PLOT_PATH="$OUT_DIR/${MODEL_NAME}_training_dynamics.png"
DELTA_PATH="$OUT_DIR/${MODEL_NAME}_delta_cone.csv"
REGIME_PATH="$OUT_DIR/${MODEL_NAME}_bad_mode_summary.json"
BASIN_PATH="$OUT_DIR/${MODEL_NAME}_basin_transitions.csv"

CSV_ARGS=(
  --log "$LOG_PATH"
  --out "$CSV_PATH"
  --model-name "$MODEL_NAME"
)
if [[ -n "$SR_CSV" ]]; then
  CSV_ARGS+=(--sr-csv "$SR_CSV")
fi
python 35_lila_log_to_csv.py "${CSV_ARGS[@]}"

PLOT_ARGS=(
  --primary "$CSV_PATH"
  --primary-label "$MODEL_NAME"
  --out "$PLOT_PATH"
)
if [[ -n "$COMPARE_CSV" ]]; then
  PLOT_ARGS+=(--compare "$COMPARE_CSV" --compare-label "$COMPARE_LABEL")
fi
python 36_lila_training_dynamics.py "${PLOT_ARGS[@]}"

FEATURES=$(python3 - <<'PY' "$CSV_PATH"
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], nrows=1)
features = ["train_loss", "val_loss"]
for col in ["sr_block0", "sr_block5", "sr_block11", "cn_block0", "cn_block5", "cn_block11"]:
    if col in df.columns:
        features.append(col)
print(",".join(features))
PY
)

python 37_lila_delta_cone_analysis.py \
  --csv "$CSV_PATH" \
  --features "$FEATURES" \
  --arrow train_loss \
  --arrow-mode decreasing \
  --regime-out "$REGIME_PATH" \
  --basin-transitions-out "$BASIN_PATH" \
  --out "$DELTA_PATH"

echo "[ok] phase2 artifacts in $OUT_DIR"
