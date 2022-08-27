import numpy as np
import os
import random
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
from sklearn.preprocessing import StandardScaler

from .. import audio as Audio
from ..utils import get_alignment, average_by_duration


def prepare_align(in_dir, meta):
    with open(os.path.join(in_dir, meta), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            basename, text = parts[0], parts[3]

            basename=basename.replace('.wav','.txt')
            
            with open(os.path.join(in_dir,'wavs',basename),'w') as f1:
                f1.write(text)

def build_from_path(in_dir, out_dir, meta, hp):
    train, val = list(), list()

    # processed 데이터에 대한 standard scaler 준비
    scalers = [StandardScaler(copy=False) for _ in range(3)]	# scalers for mel, f0, energy
    n_frames = 0
    print(os.getcwd())
    with open(os.path.join(in_dir, '../../' + meta), encoding='utf-8') as f:
        for index, line in enumerate(f):
            parts = line.strip().split('|')
            # basename은 100063/12229.wav 형식
            # basename[0] : 폴더명 / basename[1] : 파일명 

            if len(parts) < 5: # transcript 형식 오류 적발
                print('Bad Trascript Processed')
                continue

            # Alignment로 Preprocessing check
            preCheck = out_dir + '/mel/{}-mel-{}.npy'.format(hp.dataset, parts[0][:-4])
            if os.path.isfile(preCheck):
                continue
            
            ## preprocessed 데이터 추출 함수 / ret은 파일 위치 및 텍스트 + mel-spectrogram
            ret = process_utterance(in_dir, out_dir, parts[0], scalers, hp)
            
            # f0가 0인 상황과 Textgrid가 누락된 상황에 학습 데이터로 추가하지 않음
            if ret is None or ret is False:
                continue
            else:
                info, n = ret
            
            # train:val 비율 29:1로 진행
            rand = random.randrange(0,30)
            if rand < 1:
                val.append(info)
            else:
                train.append(info)


            if index % 1000 == 0:
                print("Done %d" % index)

            n_frames += n
            
    # 각 데이터마다 마련한 scaler로 param_list 생성
    if not os.path.isfile(out_dir + '/mel_stat.npy'): 
        param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
        param_name_list = ['mel_stat.npy', 'f0_stat.npy', 'energy_stat.npy']
        
        # 각 Scaler 값들을 stat들로 두어 학습과 추론 시 활용 
        [np.save(os.path.join(out_dir, param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list)]

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, filename, scalers, hp):

    wav_path = os.path.join(hp.direct_dir, '{}.wav'.format(filename[:-4]))
    
    # Textgrid 디렉토리 설정
    tg_path = os.path.join(out_dir, 'textgrids', '{}.TextGrid'.format(filename[:-4])) 

    # TextGrid가 없을 경우
    if os.path.isfile(tg_path) == False:
        print(filename[:-4], 'TextGrid is Not Found.')
        return None

    # Textgrid로부터 alignments 불러오기
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files (묵음 날려버림) - textgrid 활용
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency (pyworld를 사용한 핵심 주파수 추출 - pitch contour)
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]
    

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]

    # ! 2초 미만의 오디오가 무시되는 문제 발생하여 주석처리 (없이도 성능 좋음)
    #f0, energy = remove_outlier(f0), remove_outlier(energy)

    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)
    
    # f0[f0!=0]은 f0 안에 0을 뺀 모든 것을 의미 / 즉, 전부다 0이면 에러라는 뜻
    if len(f0[f0!=0]) == 0 :
        print(filename[:-4], 'f0 error')
        return None

    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None



    # Save alignment
    if not os.path.exists(os.path.join(out_dir, 'alignment')):
        os.makedirs(os.path.join(out_dir, 'alignment'), exist_ok=True)
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, filename[:-4])
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save f0
    if not os.path.exists(os.path.join(out_dir, 'f0')):
        os.makedirs(os.path.join(out_dir, 'f0'), exist_ok=True)
    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, filename[:-4])
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    if not os.path.exists(os.path.join(out_dir, 'energy')):
        os.makedirs(os.path.join(out_dir, 'energy'), exist_ok=True)
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, filename[:-4])
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save mel
    if not os.path.exists(os.path.join(out_dir, 'mel')):
        os.makedirs(os.path.join(out_dir, 'mel'), exist_ok=True)
    
    # Save mel-spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, filename[:-4])
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
   
    # 각 Standard Scaler를 데이터마다 fit 
    mel_scaler, f0_scaler, energy_scaler = scalers
    mel_scaler.partial_fit(mel_spectrogram.T)
    f0_scaler.partial_fit(f0[f0!=0].reshape(-1, 1))
    energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

    # train.txt / val.txt에서 사용할 텍스트 모음
    return '|'.join([filename[:-4], text]), mel_spectrogram.shape[1]
