#!/bin/bash
cd FastSpeech2
python3 s3_target_wav_load.py

# FastSpeech2 preprocess
. /opt/conda/etc/profile.d/conda.sh
conda activate aligner

python3 data_preprocessing.py

conda activate base

# FastSpeech2 Training
python3 preprocess.py
python3 train.py --restore_step kaist_300000
cd ..

# HiFi-GAN Training
cd HiFiGAN
python3 train.py