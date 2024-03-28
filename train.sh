#!/bin/bash


#python train.py --sex_loss_function CE --age_loss_function CE --data_name UTKFace --model resnet50
#python train.py --sex_loss_function CE --age_loss_function Focal --data_name UTKFace --model resnet50
#python train.py --sex_loss_function CE --age_loss_function LogitAdjust --data_name UTKFace --model resnet50
#
#python train.py --sex_loss_function CE --age_loss_function CE --data_name UTKFace --model resnet18
#python train.py --sex_loss_function CE --age_loss_function Focal --data_name UTKFace --model resnet18
#python train.py --sex_loss_function CE --age_loss_function LogitAdjust --data_name UTKFace --model resnet18

python train.py --sex_loss_function CE --age_loss_function CE --data_name imdb --model resnet50
python train.py --sex_loss_function CE --age_loss_function Focal --data_name imdb --model resnet50
python train.py --sex_loss_function CE --age_loss_function LogitAdjust --data_name imdb --model resnet50


# 定义参数选项
#sex_loss_functions=("CE" "LDAM" "Focal" "LogitAdjust")
#age_loss_functions=("CE" "LDAM" "Focal" "LogitAdjust")
#
## 循环遍历所有参数组合
#for sex_loss in "${sex_loss_functions[@]}"; do
#    for age_loss in "${age_loss_functions[@]}"; do
#        echo "Running training with sex_loss=${sex_loss} and age_loss=${age_loss}"
#        python train.py --sex_loss_function $sex_loss --age_loss_function $age_loss
#    done
#done
shutdowm