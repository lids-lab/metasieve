#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
# If you don’t want wandb online:
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# PORT=29005

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

# RUN_NAME="user_badge_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-badge \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_pruned_3hops.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_badge_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"


RUN_NAME="user_badge_mpspruning_3_hops_k_50"
LOG_FILE="${RUN_NAME}.log"   
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
  --dataset rel-stack \
  --cache_dir /home/fsk2739/relgt/data \
  --task user-badge \
  --sampling_backend neighborloader \
  --batch_size 512 \
  --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_3hops.py" \
  --num_neighbors 50 \
  --num_workers 8 \
  --epochs 10 \
  --lr 0.0001 \
  --warmup_steps 10 \
  --ff_dropout 0.1 \
  --attn_dropout 0.1 \
  --run_name "$RUN_NAME" \
  --out_dir results/rel-stack/user_badge_mpspruning_3_hops_k_50 \
  --log_file "$LOG_FILE"


# RUN_NAME="user_engagement_random_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-engagement \
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
#   --out_dir results/user_engagement_random_3_hops \
#   --log_file "$LOG_FILE"

# RUN_NAME="user_engagement_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-engagement \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-engagement/user_engagement_pruned_3hops.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_engagement_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_engagement_mpspruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-engagement \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-engagement/user_engagement_3hops.py" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_engagement_mpspruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"


# RUN_NAME="post_votes_random_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task post-votes \
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
#   --out_dir results/post_votes_random_3_hops \
#   --log_file "$LOG_FILE"


# RUN_NAME="post_votes_pruning_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task post-votes \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/post-votes/post_votes_pruned.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/post_votes_pruning_3_hops \
#   --log_file "$LOG_FILE"



# RUN_NAME="driver_dnf_random_2_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 40 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_random_2_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_pruning_2_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_2hops.json" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_pruning_2_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_mpspruning_2_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_2hops.py" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_mpspruning_2_hops_k_100 \
  # --log_file "$LOG_FILE"


# RUN_NAME="driver_position_random_2_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 40 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_random_2_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_position_pruning_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 15 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_pruning_3_hops \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_random_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 15 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_random_3_hops \
#   --log_file "$LOG_FILE"

# RUN_NAME="driver_top3_pruning_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 15 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_pruning_3_hops \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_mpspruning_3_hops"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_3hops.py" \
#   --num_neighbors 100 \
#   --max_steps_per_epoch 4000 \
#   --num_workers 8 \
#   --epochs 15 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_mpspruning_3_hops \
#   --log_file "$LOG_FILE"





