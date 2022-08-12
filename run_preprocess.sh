#!/bin/sh
cd FastSpeech2
python3 s3_target_wav_load.py

# FastSpeech2 preprocess
python3 data_preprocessing.py