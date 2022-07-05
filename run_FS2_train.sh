#!/bin/sh

echo "Enter the dataset name"
read dataset
echo "Selected Dataset: $dataset"

cd FastSpeech2

mv ../dataset/$dataset ../dataset/_$dataset

python3 preprocess.py

python3 train.py --restore_step 250000

mv ../dataset/_$dataset ../dataset/$dataset