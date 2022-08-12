import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hparams as hp
import os

import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=hp.train_visible_devices

import numpy as np
import argparse
import time
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils
import audio as Audio
import re

def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    
    # Get dataset
    dataset = Dataset("train.txt") 
    
    # shuffle=True - 인덱스 랜덤 선별 (batch의 제곱만큼 준비하고 2중 for문)
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, 
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # 학습 시 Multi / Single 판단 (Add)
    _, speaker_table = utils.get_speakers()
    print('\nSpeaker Count', len(speaker_table))

    # FastSpeech2 정의
    model = nn.DataParallel(FastSpeech2()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, int(re.sub(r'[^0-9]', '', args.restore_step)))
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            'ckpt', 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # read params
    """mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)"""


    # Load HiFi-GAN or VocGAN
    # Fine-Tune은 Evaluate을 수행하지 않음
    """if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path = hp.vocoder_pretrained_model_path)
        vocoder.to(device)
    elif hp.vocoder == 'hifigan':
        vocoder = utils.get_hifigan(ckpt_path = hp.vocoder_pretrained_model_path)
        vocoder.to(device)
    else:
        vocoder = None"""
    vocoder = None

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    # Training
    model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        # loader를 불러 getitems 함수 수행
        for i, batchs in enumerate(loader):
            # batch가 8인 경우 64개 중에 8개 선별
            for j, data_of_batch in enumerate(batchs):
                # metadata가 잘못 작성된 데이터가 있을 경우 배치 단위 생략
                if type(data_of_batch) == bool:
                    continue

                start_time = time.perf_counter()
                current_step = i*hp.batch_size + j + int(re.sub(r'[^0-9]', '', args.restore_step)) + epoch*len(loader)*hp.batch_size + 1

                # Speaker Embedding을 위한 Embedding ID
                speaker_ids = []
                for t in data_of_batch["id"]:
                    speaker_ids.append(t)
                

                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                
                # Forward
                # Speaker ID를 추가하여 Multi-Speaker FastSpeech2 구현
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                    text, src_len, speaker_ids, mel_len, D, f0, energy, max_src_len, max_mel_len)
                
                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
                 
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
                with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")
                with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")
                with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")
                with open(os.path.join(log_path, "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")
                with open(os.path.join(log_path, "f0_loss.txt"), "a") as f_f_loss:
                    f_f_loss.write(str(f_l)+"\n")
                with open(os.path.join(log_path, "energy_loss.txt"), "a") as f_e_loss:
                    f_e_loss.write(str(e_l)+"\n")
                 
                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "Time Used: {:.3f}s({:.1f}min), Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), ((Now-Start)/60), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    
                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                train_logger.add_scalar('Loss/total_loss', t_l, current_step)
                train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                train_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                train_logger.add_scalar('Loss/energy_loss', e_l, current_step)
                
                if current_step % hp.save_step == 0:
                    if not os.path.exists(os.path.join(checkpoint_path, hp.user_id)):
                        os.mkdir(os.path.join(checkpoint_path, hp.user_id))
                    if not os.path.exists(os.path.join(checkpoint_path, hp.target_dir)):
                        os.mkdir(os.path.join(checkpoint_path, hp.target_dir))

                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, hp.target_dir, 'checkpoint_{}_{}.pth'.format(hp.dataset, current_step)))
                    print("save model at step {} ...".format(current_step))
                    
                    print(datetime.datetime.now() + datetime.timedelta(hours=9))

                    
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=str, default='250000')
    args = parser.parse_args()

    main(args)
