#!/bin/bash

python train.py --model efficientnet --epochs 1 --device cuda:0 &
python train.py --model resnet --epochs 1 --device cuda:1 &

wait

python train.py --model convnext --epochs 1 --device cuda:0 &
python train.py --model vit --epochs 1 --device cuda:1 &

