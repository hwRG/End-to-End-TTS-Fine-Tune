import torch
import torch.nn as nn
import numpy as np
import os

from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from param import user_param

import argparse
import re
from string import punctuation

from .fastspeech2 import FastSpeech2

from .text import text_to_sequence
from . import utils

from .g2pk.g2pk import G2p
from jamo import h2j

from . import hparams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Synthesizer:
    def __init__(self, hp):
        self.hp = hp

        self.total_mel_postnet_torch = []
        self.total_f0_output = []
        self.total_energy_output = []

        self.mean_mel, self.std_mel = torch.tensor(np.load(os.path.join(self.hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
        self.mean_f0, self.std_f0 = torch.tensor(np.load(os.path.join(self.hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
        self.mean_energy, self.std_energy = torch.tensor(np.load(os.path.join(self.hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

        self.mean_mel, self.std_mel = self.mean_mel.reshape(1, -1), self.std_mel.reshape(1, -1)
        self.mean_f0, self.std_f0 = self.mean_f0.reshape(1, -1), self.std_f0.reshape(1, -1)
        self.mean_energy, self.std_energy = self.mean_energy.reshape(1, -1), self.std_energy.reshape(1, -1)

        
    def kor_preprocess(self, text):
        text = text.rstrip(punctuation)
        # h2j와 seq 변환 항목만 남김
        phone = h2j(text)
        print('after h2j: ',phone)
        phone = list(filter(lambda p: p != ' ', phone))
        phone = '{' + '}{'.join(phone) + '}'
        print('phone: ',phone)
        phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
        print('after re.sub: ',phone)
        phone = phone.replace('}{', ' ')

        print('|' + phone + '|')
        sequence = np.array(text_to_sequence(phone, self.hp.text_cleaners))
        sequence = np.stack([sequence])

        return torch.from_numpy(sequence).long().to(device)


    def get_FastSpeech2(self):
        checkpoint_path = os.path.join(self.hp.checkpoint_path, self.hp.target_dir, "checkpoint_{}_{}.pth".format(self.hp.dataset, self.hp.synthesize_step))
        
        model = nn.DataParallel(FastSpeech2())
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
        model.requires_grad = False
        model.eval()

        return model


    def fastspeech2_inference(self, model, text):
        text = self.kor_preprocess(text)
        src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

        target_voice = os.path.join(self.hp.direct_dir, '77.wav')

        mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, synthesize=True, target_voice=target_voice)
        
        mel_torch = mel.transpose(1, 2).detach()
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
        f0_output = f0_output[0]
        energy_output = energy_output[0]

        mel_torch = utils.de_norm(mel_torch.transpose(1, 2), self.mean_mel, self.std_mel)
        mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), self.mean_mel, self.std_mel).transpose(1, 2)
        f0_output = utils.de_norm(f0_output, self.mean_f0, self.std_f0).squeeze().detach().cpu().numpy()
        energy_output = utils.de_norm(energy_output, self.mean_energy, self.std_energy).squeeze().detach().cpu().numpy()
        
        self.total_mel_postnet_torch.append(mel_postnet_torch)
        self.total_f0_output.append(f0_output)
        self.total_energy_output.append(energy_output)


    def synthesize(self, text):
        now = datetime.now()
        time = now.strftime('%Y-%m-%d_%H.%M.%S')

        self.model = self.get_FastSpeech2().to(device)

        sentence = text
        sentence = sentence.split(' ')[:6]
        sent = ''
        for i in sentence:
            sent += i + ' '
        sentence = sent

        # 나뉘기 전에 g2p를 적용하여 숫자와 영어 한글 변환
        g2p=G2p()
        text = g2p(text) 
        print('After g2p: ',text)

        text_list = text.split(' ')
        wordCnt = 0
        stop_token = ',.?!~'
        synth_flag = False
        text = ''
        sentence_list = []
        for word in text_list:
            text += word + ' '
            wordCnt += 1

            # 나누는 기호로 단위로 끊어 생성 (,./; 등)
            for stop in stop_token:
                if word[-1] == stop:
                    text = text[:-1] # 기호 제거
                    synth_flag = True
                    break

            if synth_flag:
                sentence_list.append(text)
                wordCnt = 0
                self.fastspeech2_inference(self.model, text)
                synth_flag = False
                text = ''

        if wordCnt > 0:
            sentence_list.append(text)
            self.fastspeech2_inference(self.model, text)

        os.makedirs(self.hp.test_path, exist_ok=True)

        wav_path = os.path.join(self.hp.test_path, '{}.wav'.format(time))
        # 문장 단위로 합성
        utils.hifigan_infer(self.total_mel_postnet_torch, self.hp, path=wav_path, synthesize=True)   
        
        os.makedirs(self.hp.test_path + '/plot', exist_ok=True)
        
        return wav_path

def custom_synthesizer(id='hws0120', target='HW-man'):
    # Test
    param = user_param.UserParam(id, target)
    hp = hparams.hparam(param)
    synthesizer = Synthesizer(hp)

    # KSS 기준 train sentence
    train_sentence=['그는 괜찮은 척하려고 애쓰는 것 같았다','그녀의 사랑을 얻기 위해 애썼지만 헛수고였다','용돈을 아껴써라','그는 아내를 많이 아낀다','요즘 공부가 안돼요','한 여자가 내 옆에 앉았다']
    #train_sentence=['가까운 시일 내에 한번, 댁으로 찾아가겠습니다','우리의 승리는 기적에 가까웠다','아이들의 얼굴에는 행복한 미소가 가득했다','헬륨은 공기보다 가볍다','이것은 간단한 문제가 아니다']
    # Unseen 8문장 합성
    eval_sentence=['어제 들었는데, 윌라과제 하기로 했대. 그래서 홈페이지가서 봤더니, 벌써 올라와 있더라고. 업데이트가 진짜 빠른것 같아.',
                    '너무 졸린데, 과제 언제 다하지? 그냥 때려치울까? 안돼 그거는! 대학생으로서의 도리가 아니야. 힘내자 나자신아.',
                    '목소리 테스트하는데, 대본을 쓸게 없는거야. 그래서 어떤걸 쓸까 했는데, 지금 고민하는 대사를 쓰면 되더라고.',
                    '나는 지금 대학교 4학년이고, 졸업을 앞두고 있다. 졸업 이후에는 대학원과 취업이 있는데, 병특 때문에 어떻게할지 모르겠어.',
                    '대본을 이정도 썼으면, 테스트 하기에는 충분하겠지? 빨리 추가로 대본녹음 해야하는데, 하기가 은근히 귀찮네',
                    '있잖아 내가 물어볼게 있어. 이목소리랑 저목소리랑 둘중에 뭐가 더좋은 것같아?',
                    '관리사무소에서 안내말씀 드립니다. 매주 목요일은 아파트점검일입니다. 까스표를 보고 현관문 앞에 점검 사실을 기록해주세요. 다시한번 안내말씀 드립니다. 매주 목요일은 아파트점검일입니다. 까스표를 보고 현관문 앞에 점검 사실을 기록해주세요. 이상은 관리사무소에서 안내말씀 드렸습니다.',
                    '안녕하세요 여러분. 오늘은 갈비찜을 만들어 볼거에요. 먼저 물을 삼백미리를 넣어주고, 베이킹파우더를 두스푼 넣어주세요. 그리고 30분정도 기다렸다가, 고기를 넣고 에어프라이어에 돌려주세요. 시간은 15분이고 온도는 180도에요. 다되면 가족끼리 맛있게 먹어요. 그럼 안녕~']
    test_sentence='안녕하세요 반갑습니다 테스트랍니다'
    
    print('Which sentence do you want?')
    print('1.Experient_sentence 2.train_sentence 3.test_sentence 4.create new sentence')

    mode=input()
    print('you went for mode {}'.format(mode))
    if mode=='4':
        print('input sentence')
        sentence = input()
    elif mode=='1':
        sentence = eval_sentence
    elif mode=='2':
        sentence = train_sentence
    elif mode=='3':
        sentence = test_sentence
    else:
        exit()
    
    print('Sentence that will be synthesized: ')
    print(sentence)
    if mode == '1' or mode== '2':
        for sent in sentence:
            synthesizer.synthesize(sent)
    else:
        synthesizer.synthesize(sentence)

if __name__ == "__main__":
    custom_synthesizer()
