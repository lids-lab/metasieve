#!/usr/bin/env bash
set -euo pipefail

# Optional: pick GPU
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Global configs
DATASET_NAME="rel-ratebeer"
OUTPUT_ROOT="outputs"
TASK_NAME="user-count" # "user-badge", "user-engagement", "post-votes", "driver-dnf", "driver-top3", "driver-position"
TASK_KIND="regression" #regression #classification
NUM_LAYERS=3

EPOCHS=20
# BATCH_SIZE=512
# BATCH_SIZE=256
BATCH_SIZE=128
TEXT_EMB_BATCH_SIZE=256
EARLY_STOP_PATIENCE=10
MAX_STEPS_PER_EPOCH=1500

FANOUT=64
MODEL_TYPE="hgt"   # "hgs" or "hgt"
ATTENTION_HEAD=4
# pruning config (leave empty for random fanout)
# ./outputs/rel-stack/user-badge/user_badge_pruned.json
# ./outputs/rel-stack/user-badge/user_badge_3hops.py
# ./outputs/rel-stack/user-badge/user_badge_4hops.py

# ./outputs/rel-stack/user-engagement/user_engagement_pruned.json
# ./outputs/rel-stack/user-engagement/user_engagement_3hops.py
# ./outputs/rel-stack/user-engagement/user_engagement_4hops.py


# ./outputs/rel-stack/post-votes/post_votes_pruned.json

# ./outputs/rel-f1/driver-dnf/driver_dnf_pruned_3hops.json
# ./outputs/rel-f1/driver-dnf/driver_dnf_pruned_4hops.json
# ./outputs/rel-f1/driver-dnf/driver_dnf_pruned_2hops.json
# ./outputs/rel-f1/driver-dnf/driver_dnf_3hops.py
# ./outputs/rel-f1/driver-dnf/driver_dnf_2hops.py

# ./outputs/rel-f1/driver-top3/driver_top3_pruned_2hops.json
# ./outputs/rel-f1/driver-top3/driver_top3_pruned_3hops.json
# ./outputs/rel-f1/driver-top3/driver_top3_pruned_4hops.json
# ./outputs/rel-f1/driver-top3/driver_top3_2hops.py
# ./outputs/rel-f1/driver-top3/driver_top3_3hops.py

# ./outputs/rel-f1/driver-position/driver_position_pruned_2hops.json
# ./outputs/rel-f1/driver-position/driver_position_pruned_3hops.json
# ./outputs/rel-f1/driver-position/driver_position_pruned_4hops.json

# ./outputs/rel-avito/user-clicks/user_clicks_pruned_3hops.json
# ./outputs/rel-avito/user-visits/user_visits_pruned_3hops.json


# ./outputs/rel-f1/driver-dnf/driver_dnf_pruned_2hops.json
# ./outputs/rel-trial/study-adverse/study_adverse_pruned_3hops.json

# ./outputs/rel-ratebeer/user-churn/user_churn_pruned_3hops.json
SAMPLING_CFG_PATH="./outputs/rel-ratebeer/user-count/user_count_pruned_3hops_0.1.json"  # e.g. "./outputs/rel-stack/user-badge/user_badge_pruned_v1.json"

python train_stepwise.py \
  --device cuda:0 \
  --dataset_name "$DATASET_NAME" \
  --output_root "$OUTPUT_ROOT" \
  --task_name "$TASK_NAME" \
  --task_kind "$TASK_KIND" \
  --num_layers "$NUM_LAYERS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --text_emb_batch_size "$TEXT_EMB_BATCH_SIZE" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --max_steps_per_epoch "$MAX_STEPS_PER_EPOCH" \
  --fanout "$FANOUT" \
  --heads "$ATTENTION_HEAD"\
  --model_type "$MODEL_TYPE" \
  ${SAMPLING_CFG_PATH:+--sampling_cfg_path "$SAMPLING_CFG_PATH"}
