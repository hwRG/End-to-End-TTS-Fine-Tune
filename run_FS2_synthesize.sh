#!/bin/sh

echo "Enter the dataset name"
read dataset
echo "Selected Dataset: $dataset"

cd FastSpeech2

mv ../dataset/$dataset ../dataset/_$dataset

python3 synthesize.py --step 255000

mv ../dataset/_$dataset ../dataset/$dataset