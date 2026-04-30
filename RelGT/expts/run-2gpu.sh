#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true

PORT=29100

torchrun --nproc_per_node=2 --master_port=$PORT main_node_ddp.py \
  --dataset rel-stack \
  --cache_dir /home/fsk2739/relgt/data \
  --task user-badge \
  --sampling_backend neighborloader \
  --batch_size 512 \
  --nl_num_neighbors "64,64" \
  --num_neighbors 300 \
  --num_workers 4 \
  --epochs 10 \
  --lr 0.0001 \
  --warmup_steps 10 \
  --ff_dropout 0.1 \
  --attn_dropout 0.1 \
  --run_name two-gpu-ddp-our-sampling \
  --out_dir results/two-gpu-ddp-our-sampling
