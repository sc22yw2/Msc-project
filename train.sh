#!/bin/bash

# 定义参数选项
sex_loss_functions=("CE" "LDAM" "Focal" "LogitAdjust")
age_loss_functions=("CE" "LDAM" "Focal" "LogitAdjust")

# 循环遍历所有参数组合
for sex_loss in "${sex_loss_functions[@]}"; do
    for age_loss in "${age_loss_functions[@]}"; do
        echo "Running training with sex_loss=${sex_loss} and age_loss=${age_loss}"
        python train.py --sex_loss_function $sex_loss --age_loss_function $age_loss
    done
done
shutdowm