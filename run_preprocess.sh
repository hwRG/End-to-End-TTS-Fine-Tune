#!/bin/bash
cd FastSpeech2
python3 s3_target_wav_load.py

# FastSpeech2 preprocess
. /opt/conda/etc/profile.d/conda.sh
conda activate aligner

python3 data_preprocessing.py