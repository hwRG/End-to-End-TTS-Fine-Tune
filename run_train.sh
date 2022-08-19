#!/bin/sh

# FastSpeech2 Training
cd FastSpeech2
python3 preprocess.py
python3 train.py
cd ..

# HiFi-GAN Training
cd HiFiGAN
python3 train.py