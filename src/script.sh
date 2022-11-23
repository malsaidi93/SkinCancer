#!/bin/bash

conda activate torchcv
python train.py --model resnet --epoch 1 --device cuda:0
python train.py --model vit --epoch 1 --device cuda:1

wait
python train.py --model efficientnet --epoch 1 --device cuda:0
python train.py --model convnext --epoch 1 --device cuda:1


echo "Completed.."
