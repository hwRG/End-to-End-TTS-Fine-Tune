import torch
import torch.nn as nn
import numpy as np
import hparams as hp
import os

from utils import get_speakers


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=hp.synth_visible_devices

import argparse
import re
from string import punctuation

from fastspeech2 import FastSpeech2

from text import text_to_sequence
import utils
import audio as Audio

import codecs
from g2pk.g2pk import G2p
from jamo import h2j
from ts.torch_handler.base_handler import BaseHandler


def get_FastSpeech2(num, synthesize=False):
    checkpoint_path = os.path.join(hp.checkpoint_path, hp.dataset, "checkpoint_{}_{}.pth".format(hp.dataset, num))
    model = nn.DataParallel(FastSpeech2(synthesize=synthesize))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model


def kor_preprocess(text):
    text = text.rstrip(punctuation)
    phone = h2j(text)
    print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone,hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long()#.to(device)


class TTSHandler(BaseHandler):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset = hp.dataset
    

    def initialize(self):
        self.g2p=G2p()
        self.model = get_FastSpeech2('255000', True).to(self.device)
        
        self.mean_mel, self.std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(self.device)
        self.mean_f0, self.std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(self.device)
        self.mean_energy, self.std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(self.device)

        self.mean_mel, self.std_mel = self.mean_mel.reshape(1, -1), self.std_mel.reshape(1, -1)
        self.mean_f0, self.std_f0 = self.mean_f0.reshape(1, -1), self.std_f0.reshape(1, -1)
        self.mean_energy, self.std_energy = self.mean_energy.reshape(1, -1), self.std_energy.reshape(1, -1)

        self.initialized = True
    
    def preprocess(self, data):
        data = self.g2p(data) 

        self.text_list = data.split(' ')
        self.wordCnt = 0
        self.stop_token = ',.?!~'
        self.synth_flag = False
        data = ''
        self.total_mel_postnet_torch = []; self.total_f0_output = []; self.total_energy_output = []
        
        return data

    
    def inference(self, data):
        data = kor_preprocess(data).to(self.device)
        src_len = torch.from_numpy(np.array([data.shape[1]])).to(self.device)

        mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = self.model(data, src_len, synthesize=True)
        
        mel_torch = mel.transpose(1, 2).detach()
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
        f0_output = f0_output[0]
        energy_output = energy_output[0]

        mel_torch = utils.de_norm(mel_torch.transpose(1, 2), self.mean_mel, self.std_mel)
        mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), self.mean_mel, self.std_mel).transpose(1, 2)
        f0_output = utils.de_norm(f0_output, self.mean_f0, self.std_f0).squeeze().detach().cpu().numpy()
        energy_output = utils.de_norm(energy_output, self.mean_energy, self.std_energy).squeeze().detach().cpu().numpy()
        
        # HiFi-GAN에서 사용 할 mel
        self.total_mel_postnet_torch.append(mel_postnet_torch)
        self.total_f0_output.append(f0_output)
        self.total_energy_output.append(energy_output)
    
    """def postprocess(self, data):   
        return data.strip()"""

    def handle(self, data):
        sentence = data.split(' ')[:6]
        sent = ''
        for i in sentence:
            sent += i + ' '
        sentence = sent
        while True: # 챗봇 생성 다음 단어가 eof면 끝내도록
            # 마지막으로 
            data = self.preprocess(data)
            for word in self.text_list:
                data += word + ' '
                wordCnt += 1

                for stop in self.stop_token:
                    if word[-1] == stop:
                        data = data[:-1] # 기호 제거
                        synth_flag = True
                        break

                if synth_flag:
                    wordCnt = 0
                    self.inference(text)
                    synth_flag = False
                    text = ''

            if wordCnt > 0: # 마지막 문장
                self.inference(text)
                break
        
            utils.hifigan_infer(self.total_mel_postnet_torch, path=os.path.join(hp.test_path, '{}.wav'.format(sentence)), synthesize=True)
        # dict or array 형태로 return
        #return 
    