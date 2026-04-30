#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
# If you don’t want wandb online:
export WANDB_MODE=disabled
export WANDB_DISABLED=true


# RUN_NAME="study_outcome_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="study_outcome_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-outcome/study_outcome_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="study_outcome_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-outcome/study_outcome_random_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="study_adverse_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-adverse/study_adverse_random_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="site_success_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/site-success/site_success_random_pruned_3hops_0.2.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE" 


# RUN_NAME="driver_dnf_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_random_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE" 

# RUN_NAME="driver_position_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_random_pruned_3hops_0.1.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE" 

# RUN_NAME="driver_top3_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_random_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_visits_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_random_pruned_3hops_0.1.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_clicks_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_random_pruned_3hops_0.4.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


RUN_NAME="ad_ctr_random_pruning_3_hops_k_100"
LOG_FILE="${RUN_NAME}.log"   
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
  --dataset rel-avito \
  --cache_dir /home/fsk2739/relgt/data \
  --task ad-ctr \
  --sampling_backend neighborloader \
  --batch_size 512 \
  --nl_num_neighbors "@pruning_configs/rel-avito/ad-ctr/ad_ctr_random_pruned_3hops_0.2.json" \
  --num_neighbors 100 \
  --num_workers 8 \
  --epochs 30 \
  --lr 0.0001 \
  --warmup_steps 10 \
  --ff_dropout 0.1 \
  --attn_dropout 0.1 \
  --run_name "$RUN_NAME" \
  --out_dir results/rel-avito/ad_ctr_random_pruning_3_hops_k_100 \
  --log_file "$LOG_FILE"




# RUN_NAME="study_outcome_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-outcome/study_outcome_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"






# RUN_NAME="study_outcome_mpspruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-outcome \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-outcome/study_outcome_3hops.py" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_outcome_mpspruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="driver_dnf_no_rate_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_3hops_no_rate.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_no_rate_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_pruned_3hops_randomv2.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_dnf_mpspruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-dnf \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-dnf/driver_dnf_3hops.py" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_dnf_mpspruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_position_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_position_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="driver_position_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="driver_position_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-position \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-position/driver_position_pruned_3hops_randomv3.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_position_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="driver_top3_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="driver_top3_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_pruned_3hops_randomv2.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="driver_top3_mpspruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-f1 \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task driver-top3 \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-f1/driver-top3/driver_top3_3hops.py" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-f1/driver_top3_mpspruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_visits_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_visits_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="user_visits_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_visits_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_pruned_3hops_randomv2.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_visits_mpspruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-visits \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-visits/user_visits_3hops.py" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_visits_mpspruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_clicks_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_clicks_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_clicks_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_clicks_random_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_pruned_3hops_randomv2.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_random_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"




# RUN_NAME="user_clicks_mpspruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-clicks \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/user-clicks/user_clicks_3hops.py" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/user_clicks_mpspruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_churn_random_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-churn \
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
#   --out_dir results/rel-ratebeer/user_churn_random_3_hops_k_50 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_churn_mpspruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-churn \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-ratebeer/user-churn/user_churn_3hops.py" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/user_churn_mpspruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_churn_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-churn \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-ratebeer/user-churn/user_churn_pruned_3hops_0.1.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/user_churn_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"



# RUN_NAME="beer_churn_random_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task beer-churn \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "16,16,16" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/beer_churn_random_3_hops_k_50_fanout16 \
#   --resume_ckpt results/rel-ratebeer/beer_churn_random_3_hops_k_50_fanout16/rel-ratebeer/beer-churn/finetuned.pt \
#   --log_file "$LOG_FILE"


# RUN_NAME="beer_churn_mpspruning_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task beer-churn \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-ratebeer/beer-churn/beer_churn_3hops_fanout16.py" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/beer_churn_mpspruning_3_hops_k_50_fanout16 \
#   --log_file "$LOG_FILE"


# RUN_NAME="beer_churn_pruning_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task beer-churn \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-ratebeer/beer-churn/beer_churn_pruned_3hops_0.1_fanout16.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/beer_churn_pruning_3_hops_k_50_fanout16 \
#   --log_file "$LOG_FILE"




# RUN_NAME="post_votes_random_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task post-votes \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "16,16,16" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/post_votes_random_3_hops_k_50_fanout16 \
#   --log_file "$LOG_FILE"


# RUN_NAME="post_votes_pruning_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task post-votes \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/post-votes/post_votes_pruned_fanout16.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/post_votes_pruning_3_hops_k_50_fanout16 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_count_random_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-count \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "16,16,16" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/user_count_random_3_hops_k_50_fanout16 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_count_pruning_3_hops_k_50_fanout16"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-ratebeer \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-count \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-ratebeer/user-count/user_count_pruned_3hops_0.1_fanout16.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-ratebeer/user_count_pruning_3_hops_k_50_fanout16 \
#   --log_file "$LOG_FILE"



# RUN_NAME="user_badge_random_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-badge \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_pruned_3hops_randomv2.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_badge_random_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"

# RUN_NAME="user_badge_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-badge \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_pruned_3hops_0.1.json" \
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




# RUN_NAME="user_badge_no_coverage_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-badge \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-badge/user_badge_pruned_3hops_no_coverage.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_badge_no_coverage_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"




# RUN_NAME="user_engagement_random_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-engagement \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-engagement/user_engagement_pruned_3hops_randomv2.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_engagement_random_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"


# RUN_NAME="user_engagement_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-engagement \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-engagement/user_engagement_pruned_3hops_0.4.json" \
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


# RUN_NAME="user_engagement_no_coverage_pruning_3_hops_k_50"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-stack \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task user-engagement \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-stack/user-engagement/user_engagement_pruned_3hops_no_coverage.json" \
#   --num_neighbors 50 \
#   --num_workers 8 \
#   --epochs 10 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-stack/user_engagement_no_coverage_pruning_3_hops_k_50 \
#   --log_file "$LOG_FILE"





# RUN_NAME="study_adverse_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="study_adverse_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-adverse/study_adverse_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="study_adverse_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-adverse/study_adverse_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="study_adverse_pruning_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task study-adverse \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/study-adverse/study_adverse_pruned_3hops_randomv3.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/study_adverse_pruning_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="site_success_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="site_success_pruning_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/site-success/site_success_pruned_3hops_randomv3.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_pruning_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="site_success_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/site-success/site_success_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"

# RUN_NAME="site_success_no_rate_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-trial \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task site-success \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-trial/site-success/site_success_pruned_3hops_no_rate.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-trial/site_success_no_rate_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="ad_ctr_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task ad-ctr \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "64,64,64" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/ad_ctr_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="ad_ctr_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task ad-ctr \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/ad-ctr/ad_ctr_pruned_3hops.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/ad_ctr_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"


# RUN_NAME="ad_ctr_no_coverage_pruning_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task ad-ctr \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/ad-ctr/ad_ctr_pruned_3hops_no_coverage.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/ad_ctr_no_coverage_pruning_3_hops_k_100 \
#   --log_file "$LOG_FILE"



# RUN_NAME="ad_ctr_pruning_random_3_hops_k_100"
# LOG_FILE="${RUN_NAME}.log"   
# torchrun --standalone --nnodes=1 --nproc_per_node=1 main_node_ddp_v2.py \
#   --dataset rel-avito \
#   --cache_dir /home/fsk2739/relgt/data \
#   --task ad-ctr \
#   --sampling_backend neighborloader \
#   --batch_size 512 \
#   --nl_num_neighbors "@pruning_configs/rel-avito/ad-ctr/ad_ctr_pruned_3hops_randomv3.json" \
#   --num_neighbors 100 \
#   --num_workers 8 \
#   --epochs 30 \
#   --lr 0.0001 \
#   --warmup_steps 10 \
#   --ff_dropout 0.1 \
#   --attn_dropout 0.1 \
#   --run_name "$RUN_NAME" \
#   --out_dir results/rel-avito/ad_ctr_pruning_random_3_hops_k_100 \
#   --log_file "$LOG_FILE"