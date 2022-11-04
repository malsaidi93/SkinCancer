#!/bin/bash
url='https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/download'
file='HAM10k'

mkdir $file
cd $file

echo "DOownloading"
curl -L $url -o $file && unzip -q $file -d $file && rm $file