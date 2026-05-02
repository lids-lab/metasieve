#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
# If you don’t want wandb online:
export WANDB_MODE=disabled
export WANDB_DISABLED=true


# RUN_NAME="user_badge_random_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-badge \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/user_badge_random_3_hops \
#   --log_file "$LOG_FILE"

RUN_NAME="user_badge_pruning_3_hops_k_50"
LOG_FILE="${RUN_NAME}.log"   
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
  --dataset rel-stack \
  --cache_dir /home/fsk2739/relgt/data \
  --task user-badge \
  --sampling_backend neighborloader \
  --batch_size 512 \
  --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_pruned_3hops.json" \
  --num_neighbors 50 \
  --num_workers 8 \
  --epochs 10 \
  --lr 0.0001 \
  --warmup_steps 10 \
  --ff_dropout 0.1 \
  --attn_dropout 0.1 \
  --run_name "$RUN_NAME" \
  --out_dir results/rel-stack/user_badge_pruning_3_hops_k_50 \
  --log_file "$LOG_FILE"


# RUN_NAME="user_badge_mpspruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-badge \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_3hops.py" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_badge_mpspruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"


