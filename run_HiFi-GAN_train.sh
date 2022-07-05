#!/bin/sh

echo "Enter the dataset name"
read dataset
echo "Selected Dataset: $dataset"

cd HiFi-GAN

mv ../dataset/$dataset ../dataset/_$dataset

python3 train.py

mv ../dataset/_$dataset ../dataset/$dataset
