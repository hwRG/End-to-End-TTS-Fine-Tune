from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from vocoder.hifigan_generator import Generator
import hparams as hp

MAX_WAV_VALUE = 32768.0
h = None
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def inference(mel, path):
    h = hp

    generator = Generator().to(device)

    #state_dict_g = load_checkpoint(h.vocoder_pretrained_model_path, device)
    state_dict_g = load_checkpoint('vocoder/pretrained_models/g_00790000.pt', device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        x = mel
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        write(path, h.sampling_rate, audio)
        print(path)

