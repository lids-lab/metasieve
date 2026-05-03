#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Pipeline configuration — edit these
# ============================================================
DATASET="rel-stack"
TASK="user-badge"
TASK_KIND="classification"    # classification | regression

# LAYERS controls both the search depth and the config hop limit
LAYERS=3

# Config generation
K=20              # top-K paths to include in sampling config
FANOUT=64         # neighbors sampled per edge type per hop

# Model / graph
CHANNELS=256
NUM_NEIGHBORS=32
TEXT_EMB_BATCH_SIZE=256

# Training sample sizes (for search scoring)
N_POS_TRAIN=250
N_NEG_TRAIN=250
N_POS_VAL=60
N_NEG_VAL=60

# Search hyperparameters
SEARCH_K=3
SEARCH_BEAM_WIDTH=6
SEARCH_SAMPLING_SIZE=1
SEARCH_RANK_METRIC="auc_roc"

# Paths
DATA_DIR="./data"
DEVICE="cuda:0"

# ============================================================
# Derived paths — no need to edit below
# ============================================================
TASK_PREFIX="${TASK//-/_}"
RESULTS_DIR="./search_results/${DATASET}"
LOG_PATH="${RESULTS_DIR}/mps_search_${TASK}.log"
TXT_PATH="${RESULTS_DIR}/mps_search_${TASK}.txt"
DECODED_PATH="${RESULTS_DIR}/mps_search_${TASK}_decoded.log"
CONFIG_OUT="${RESULTS_DIR}/${TASK_PREFIX}_${LAYERS}hops.py"

# ============================================================
# Step 1: MPS metapath search
# ============================================================
echo "=========================================="
echo "[Step 1] MPS metapath search"
echo "  Dataset : ${DATASET} / ${TASK}"
echo "  Layers  : ${LAYERS}"
echo "  Log     : ${LOG_PATH}"
echo "=========================================="

python run_search.py \
    --dataset              "$DATASET" \
    --task                 "$TASK" \
    --task_kind            "$TASK_KIND" \
    --channels             "$CHANNELS" \
    --num_layers           "$LAYERS" \
    --num_neighbors        "$NUM_NEIGHBORS" \
    --text_emb_batch_size  "$TEXT_EMB_BATCH_SIZE" \
    --n_pos_train          "$N_POS_TRAIN" \
    --n_neg_train          "$N_NEG_TRAIN" \
    --n_pos_val            "$N_POS_VAL" \
    --n_neg_val            "$N_NEG_VAL" \
    --search_l_max         "$LAYERS" \
    --search_k             "$SEARCH_K" \
    --search_beam_width    "$SEARCH_BEAM_WIDTH" \
    --search_sampling_size "$SEARCH_SAMPLING_SIZE" \
    --search_rank_metric   "$SEARCH_RANK_METRIC" \
    --device               "$DEVICE" \
    --data_dir             "$DATA_DIR"

# ============================================================
# Step 2: Decode search results
# ============================================================
echo ""
echo "=========================================="
echo "[Step 2] Decoding search results"
echo "  Input   : ${TXT_PATH}"
echo "  Output  : ${DECODED_PATH}"
echo "=========================================="

python decode_search_results.py \
    --dataset             "$DATASET" \
    --task                "$TASK" \
    --channels            "$CHANNELS" \
    --text_emb_batch_size "$TEXT_EMB_BATCH_SIZE" \
    --device              "$DEVICE" \
    --data_dir            "$DATA_DIR"

# ============================================================
# Step 3: Generate sampling config
# ============================================================
echo ""
echo "=========================================="
echo "[Step 3] Generating sampling config"
echo "  Input   : ${DECODED_PATH}"
echo "  Top-K   : ${K}  |  Layers: ${LAYERS}  |  Fanout: ${FANOUT}"
echo "  Output  : ${CONFIG_OUT}"
echo "=========================================="

python generate_sampling_config.py \
    --log    "$DECODED_PATH" \
    --k      "$K" \
    --fanout "$FANOUT" \
    --layers "$LAYERS" \
    --out    "$CONFIG_OUT"

echo ""
echo "=========================================="
echo "Pipeline complete."
echo "  Search log   : ${LOG_PATH}"
echo "  Ranked paths : ${TXT_PATH}"
echo "  Decoded log  : ${DECODED_PATH}"
echo "  Sampling cfg : ${CONFIG_OUT}"
echo "=========================================="
