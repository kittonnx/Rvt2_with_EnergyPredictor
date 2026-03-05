#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# DDP起動（xvfbは1回でOK。子プロセスがDISPLAYを継承）
xvfb-run --auto-servernum python train.py \
  --exp_cfg_path configs/rvt2.yaml \
  --mvt_cfg_path mvt/configs/rvt2.yaml \
  --device 0,1,2,3 \
  --log-dir runs/test-otani-0218



  # --log-dir runs/TF-freeze-fuse-lambda-100
  # --log-dir runs/TF-freeze-comp-fuse-lambda-100



# --log-dir runs/TF-comp-fuse-lambda-10
# --log-dir runs/TF-comp-fuse-detach-lambda-20
# --log-dir runs/TF-alltask-lambda-7


# --log-dir runs/TF-fuse-detach-alltask-lambda-9
# --log-dir runs/notfused7111216-alltask-lambda-20
# --log-dir runs/truely-clamp-finetune_fused7111216-alltask-lambda-2.5
