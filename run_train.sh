#!/bin/sh

# FastSpeech2 Training
cd FastSpeech2
python3 preprocess.py
python3 train.py --restore_step kaist_300000
cd ..

# HiFi-GAN Training
cd HiFi-GAN
python3 train.py