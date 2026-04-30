#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
# If you don’t want wandb online:
export WANDB_MODE=disabled
export WANDB_DISABLED=true


# RUN_NAME="study_outcome_pruning_delta_ablation_0.1_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-outcome/study_outcome_pruned_3hops_0.1.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_pruning_delta_ablation_0.1_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="study_outcome_pruning_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-outcome/study_outcome_pruned_3hops_0.4_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_pruning_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="study_adverse_pruning_delta_ablation_0.4_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-adverse/study_adverse_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_pruning_delta_ablation_0.4_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="study_adverse_pruning_Q_ablation_only_rate_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-adverse/study_adverse_pruned_3hops_0.4_only_rate.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_pruning_Q_ablation_only_rate_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="site_success_pruning_delta_ablation_0.05_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/site-success/site_success_pruned_3hops_0.05.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_pruning_delta_ablation_0.05_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="site_success_pruning_delta_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/site-success/site_success_pruned_3hops_0.2_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_pruning_delta_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_pruning_delta_ablation_0.4_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_pruning_delta_ablation_0.4_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="driver_dnf_pruning_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_3hops_0.4_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_pruning_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_pruning_delta_ablation_0.4_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_pruning_delta_ablation_0.4_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_pruning_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_pruned_3hops_0.4_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_pruning_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_position_pruning_delta_ablation_0.1_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_pruned_3hops_0.1.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_pruning_delta_ablation_0.1_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="driver_position_pruning_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_pruned_3hops_0.1_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_pruning_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_visits_pruning_delta_ablation_0.4_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_pruning_delta_ablation_0.4_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="user_visits_pruning_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_pruned_3hops_0.1_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_pruning_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="user_clicks_pruning_delta_ablation_0.4_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_pruning_delta_ablation_0.4_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="user_clicks_pruning_Q_ablation_only_coverage_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_pruned_3hops_0.4_only_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_pruning_Q_ablation_only_coverage_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="ad_ctr_pruning_delta_ablation_0.4_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task ad-ctr \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/ad-ctr/ad_ctr_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/ad_ctr_pruning_delta_ablation_0.4_3_hops_k_100 \
#   --log_file "$LOG_FILE"


RUN_NAME="ad_ctr_pruning_Q_ablation_only_coverage_3_hops_k_100"
LOG_FILE="${RUN_NAME}.log"   
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
  --dataset rel-avito \
  --cache_dir /home/fsk2739/relgt/data \
  --task ad-ctr \
  --sampling_backend neighborloader \
  --batch_size 512 \
  --nl_num_neighbors "@pruning_configs/rel-avito/ad-ctr/ad_ctr_pruned_3hops_0.2_only_coverage.json" \
  --num_neighbors 100 \
  --num_workers 8 \
  --epochs 30 \
  --lr 0.0001 \
  --warmup_steps 10 \
  --ff_dropout 0.1 \
  --attn_dropout 0.1 \
  --run_name "$RUN_NAME" \
  --out_dir results/rel-avito/ad_ctr_pruning_Q_ablation_only_coverage_3_hops_k_100 \
  --log_file "$LOG_FILE"