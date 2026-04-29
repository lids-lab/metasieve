#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pipeline configuration — edit these
# ============================================================
DATASET="rel-stack"
TASK="user-badge"
DELTA=0.1
HOPS_SUFFIX="3hops"
NUM_LAYERS=3
BASE_NEIGHBORS=64
MAX_HOP=3
HOPS_TO_EXPORT="1 2"
N_BATCHES=32

# Workers
AGG_WORKERS=4
SCORE_WORKERS=16

# Label mode: classification, regression
LABEL_MODE="classification"

# Derived paths
OUT_DIR="./outputs/${DATASET}/${TASK}"
TIMING_FILE="${OUT_DIR}/${DATASET}_${TASK}_pipeline_timing.txt"

# ============================================================
# Setup
# ============================================================
mkdir -p "${OUT_DIR}"

# Clear / initialize timing file
cat > "${TIMING_FILE}" << EOF
==============================================
Pipeline Timing Report
Dataset : ${DATASET}
Task    : ${TASK}
Date    : $(date '+%Y-%m-%d %H:%M:%S')
==============================================

Configuration:
  DELTA           = ${DELTA}
  HOPS_SUFFIX     = ${HOPS_SUFFIX}
  NUM_LAYERS      = ${NUM_LAYERS}
  BASE_NEIGHBORS  = ${BASE_NEIGHBORS}
  MAX_HOP         = ${MAX_HOP}
  HOPS_TO_EXPORT  = ${HOPS_TO_EXPORT}
  N_BATCHES       = ${N_BATCHES}
  AGG_WORKERS     = ${AGG_WORKERS}
  SCORE_WORKERS   = ${SCORE_WORKERS}
  LABEL_MODE      = ${LABEL_MODE}

----------------------------------------------
Step Timings:
----------------------------------------------
EOF

echo "=========================================="
echo "Pipeline: ${DATASET} / ${TASK}"
echo "Timing file: ${TIMING_FILE}"
echo "=========================================="

PIPELINE_START=$SECONDS

# ----------------------------------------------------------
# Step 1: Aggregate generation
# ----------------------------------------------------------
echo ""
echo "[Step 1] Aggregate statistics"
STEP_START=$SECONDS

python generate_statistics.py \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --num-workers ${AGG_WORKERS} \
    --max-hop ${MAX_HOP} \
    --hops-to-export ${HOPS_TO_EXPORT} \
    --n-batches ${N_BATCHES}

STEP_ELAPSED=$(( SECONDS - STEP_START ))
echo "[Step 1] Statistics generation: ${STEP_ELAPSED}s"
echo "Step 1 - Statistics generation:        ${STEP_ELAPSED}s" >> "${TIMING_FILE}"

# ----------------------------------------------------------
# Step 2: Score generation (log_rate + log_count in one run)
# ----------------------------------------------------------
echo ""
echo "[Step 2] Normalized MI generation"
STEP_START=$SECONDS

python compute_nmi_batches.py \
    --out-dir "${OUT_DIR}" \
    --hops ${HOPS_TO_EXPORT} \
    --num-workers ${SCORE_WORKERS} \
    --label-mode "${LABEL_MODE}" \
    --hops-suffix "${HOPS_SUFFIX}" \
    --cost-nonzero-only

STEP_ELAPSED=$(( SECONDS - STEP_START ))
echo "[Step 2] NMI generation: ${STEP_ELAPSED}s"
echo "Step 2 - NMI generation:             ${STEP_ELAPSED}s" >> "${TIMING_FILE}"

# ----------------------------------------------------------
# Step 3: LCB generation
# ----------------------------------------------------------
echo ""
echo "[Step 3] LCB generation"
STEP_START=$SECONDS

python lcb_score_generation.py \
    --out-dir "${OUT_DIR}" \
    --delta ${DELTA} \
    --hops-suffix "${HOPS_SUFFIX}"

STEP_ELAPSED=$(( SECONDS - STEP_START ))
echo "[Step 3] LCB generation: ${STEP_ELAPSED}s"
echo "Step 3 - LCB generation:              ${STEP_ELAPSED}s" >> "${TIMING_FILE}"

# ----------------------------------------------------------
# Step 4: Q-score + GMM clustering
# ----------------------------------------------------------
echo ""
echo "[Step 4] Q-score + GMM clustering"
STEP_START=$SECONDS

python clustering_candidates.py \
    --out-dir "${OUT_DIR}" \
    --delta ${DELTA} \
    --hops-suffix "${HOPS_SUFFIX}"

STEP_ELAPSED=$(( SECONDS - STEP_START ))
echo "[Step 4] Q-score + GMM clustering: ${STEP_ELAPSED}s"
echo "Step 4 - Q-score + GMM clustering:    ${STEP_ELAPSED}s" >> "${TIMING_FILE}"

# ----------------------------------------------------------
# Step 5: Rule generation
# ----------------------------------------------------------
echo ""
echo "[Step 5] Rule generation"
STEP_START=$SECONDS

python generate_rules.py \
    --out-dir "${OUT_DIR}" \
    --delta ${DELTA} \
    --hops-suffix "${HOPS_SUFFIX}"

STEP_ELAPSED=$(( SECONDS - STEP_START ))
echo "[Step 5] Rule generation: ${STEP_ELAPSED}s"
echo "Step 5 - Rule generation:             ${STEP_ELAPSED}s" >> "${TIMING_FILE}"

# ----------------------------------------------------------
# Step 6: Sampling config (JSON)
# ----------------------------------------------------------
echo ""
echo "[Step 6] Sampling config (JSON)"
STEP_START=$SECONDS

python generate_sampling_config.py \
    --out-dir "${OUT_DIR}" \
    --delta ${DELTA} \
    --hops-suffix "${HOPS_SUFFIX}" \
    --num-layers ${NUM_LAYERS} \
    --base-neighbors ${BASE_NEIGHBORS} \
    --task-name "${TASK}"

STEP_ELAPSED=$(( SECONDS - STEP_START ))
echo "[Step 6] Sampling config: ${STEP_ELAPSED}s"
echo "Step 6 - Sampling config (JSON):      ${STEP_ELAPSED}s" >> "${TIMING_FILE}"

# ----------------------------------------------------------
# Total
# ----------------------------------------------------------
PIPELINE_ELAPSED=$(( SECONDS - PIPELINE_START ))

cat >> "${TIMING_FILE}" << EOF

----------------------------------------------
TOTAL PIPELINE TIME:                    ${PIPELINE_ELAPSED}s
----------------------------------------------
EOF

echo ""
echo "=========================================="
echo "Pipeline complete in ${PIPELINE_ELAPSED}s"
echo "Timing saved to: ${TIMING_FILE}"
echo "=========================================="

# Print the timing file to console as well
echo ""
cat "${TIMING_FILE}"