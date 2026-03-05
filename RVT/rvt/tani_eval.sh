#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

task_no="16"
xvfb-run --auto-servernum python eval.py --model-folder runs/rvt2  --eval-datafolder ./data/task_${task_no} --tasks all --log-name rvt_fused_feat_task${task_no} \
         --eval-episodes 500 --device 0 --headless --model-name model_99.pth

#python eval.py --model-folder runs/rvt2  --eval-datafolder ./data/test --tasks all --log-name tani --eval-episodes 25 --device 0 --save-video --model-name model_99.pth

# python eval.py --model-folder runs/rvt2  --eval-datafolder ./data/test --tasks all --log-name tani_ene_lower_plus --eval-episodes 25 --device 0  --save-video --model-name model_99.pth
# xvfb-run --auto-servernum python eval.py --model-folder runs/rvt2  --eval-datafolder ./data/test1 --tasks all --log-name juutyoutanini2 --eval-episodes 100 --device 0 --headless --save-video --model-name model_99.pth
