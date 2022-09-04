#!/bin/sh

# FastSpeech2 Training
cd FastSpeech2
python3 -m preprocess.py
python3 -m train.py
cd ..

# HiFi-GAN Training
cd HiFiGAN
python3 train.py