#!/bin/sh

echo "Enter the dataset name"
read dataset
echo "Selected Dataset: $dataset"

cd FastSpeech2

mv ../dataset/$dataset ../dataset/_$dataset

python3 data_preprocessing.py

mv ../dataset/_$dataset ../dataset/$dataset
