#!/bin/bash
set -e
echo "切换到src目录..."
cd ../src || exit 1
echo "开始训练..."
python train.py
echo "训练完成，开始消融实验..."
python ablation_study.py
echo "所有任务完成！"