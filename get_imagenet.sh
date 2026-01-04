#!/bin/bash
#
# script to fully prepare ImageNet dataset

## 1. Download the data
# get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# get devkit
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

## 2. Extract the training data:
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

## 3. Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


## 4. Delete corrupted image
# there is one png under JPEG name. some readers fail on this image so need to remove it
# this line is commented by default to avoid such unexpected behaviour
# rm train/n04266014/n04266014_10835.JPEG

## 5. Sometimes TFRecords may be usefull
# wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
