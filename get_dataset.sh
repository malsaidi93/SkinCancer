#!/bin/bash
url='https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/download'
file='HAM10k'
dir='skincancer'
mkdir $dir
cd $dir

echo "DOownloading"
curl -L $url -o $file && unzip -q $file -d $dir && rm $file