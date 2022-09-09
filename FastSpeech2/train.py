import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import time
import re
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from param import user_param

from .hparams import hparam
from .fastspeech2 import FastSpeech2
from .loss import FastSpeech2Loss
from .dataset import Dataset
from .optimizer import ScheduledOptim
from . import utils

class FS2Train:
    def __init__(self, hp):
        torch.manual_seed(0)
        # Get device
        self.hp = hp
        self.device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

        # 학습 시 Multi / Single 판단 (Add)
        """_, self.speaker_table = utils.get_speakers()
        print('\nSpeaker Count', len(self.speaker_table))"""

        # FastSpeech2 정의
        self.model = nn.DataParallel(FastSpeech2()).to(self.device)
        print("Model Has Been Defined")
        self.num_param = utils.get_param_num(self.model)
        print('Number of FastSpeech2 Parameters:', self.num_param)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=hp.betas, eps=self.hp.eps, weight_decay = self.hp.weight_decay)
        self.scheduled_optim = ScheduledOptim(self.optimizer, self.hp.decoder_hidden, self.hp.n_warm_up_step, int(re.sub(r'[^0-9]', '', self.hp.restore_step)))
        self.Loss = FastSpeech2Loss().to(self.device) 
        print("Optimizer and Loss Function Defined.")


    def load_data(self):
        # Get dataset
        self.dataset = Dataset(self.hp, "train.txt") 
        
        # shuffle=True - 인덱스 랜덤 선별 (batch의 제곱만큼 준비하고 2중 for문)
        self.loader = DataLoader(self.dataset, batch_size=self.hp.batch_size**2, shuffle=True, 
            collate_fn=self.dataset.collate_fn, drop_last=True, num_workers=0)


    """def init_log(self):
        # Init logger
        self.log_path = self.hp.log_path
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'validation'), exist_ok=True)
        self.train_logger = SummaryWriter(os.path.join(self.log_path, 'train'))
        self.val_logger = SummaryWriter(os.path.join(self.log_path, 'validation'))"""


    def train(self):
        self.load_data()
        #self.init_log()

        # Load checkpoint if exists
        checkpoint_path = os.path.join(self.hp.checkpoint_path)
        try:
            checkpoint = torch.load(os.path.join('FastSpeech2',
                'ckpt', 'checkpoint_{}.pth'.format(self.hp.restore_step)))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("\n---Model Restored at Step {}---\n".format(self.hp.restore_step))
        except:
            print("\n---Start New Training---\n")
            os.makedirs(checkpoint_path, exist_ok=True)

        # Define Some Information
        Time = np.array([])
        Start = time.perf_counter()
        # Training
        self.model = self.model.train()
        for epoch in range(self.hp.epochs):
            # Get Training Loader
            total_step = self.hp.epochs * len(self.loader) * self.hp.batch_size

            # loader를 불러 getitems 함수 수행
            for i, batchs in enumerate(self.loader):
                # batch가 8인 경우 64개 중에 8개 선별
                for j, data_of_batch in enumerate(batchs):
                    # metadata가 잘못 작성된 데이터가 있을 경우 배치 단위 생략
                    if type(data_of_batch) == bool:
                        continue

                    start_time = time.perf_counter()
                    current_step = i*self.hp.batch_size + j + int(re.sub(r'[^0-9]', '', self.hp.restore_step)) + epoch*len(self.loader)*self.hp.batch_size + 1

                    # Speaker Embedding을 위한 Embedding ID
                    speaker_ids = []
                    for t in data_of_batch["id"]:
                        speaker_ids.append(t)
                    
                    text = torch.from_numpy(data_of_batch["text"]).long().to(self.device)
                    mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(self.device)
                    D = torch.from_numpy(data_of_batch["D"]).long().to(self.device)
                    log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(self.device)
                    f0 = torch.from_numpy(data_of_batch["f0"]).float().to(self.device)
                    energy = torch.from_numpy(data_of_batch["energy"]).float().to(self.device)
                    src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(self.device)
                    mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(self.device)
                    max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                    max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                    
                    # Forward
                    # Speaker ID를 추가하여 Multi-Speaker FastSpeech2 구현
                    mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = self.model(
                        text, src_len, speaker_ids, mel_len, D, f0, energy, max_src_len, max_mel_len)
                    
                    # Cal Loss
                    mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = self.Loss(
                            log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                    total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
                    
                    # Logger
                    t_l = total_loss.item()
                    m_l = mel_loss.item()
                    m_p_l = mel_postnet_loss.item()
                    d_l = d_loss.item()
                    f_l = f_loss.item()
                    e_l = e_loss.item()
                    """with open(os.path.join(self.log_path, "total_loss.txt"), "a") as f_total_loss:
                        f_total_loss.write(str(t_l)+"\n")
                    with open(os.path.join(self.log_path, "mel_loss.txt"), "a") as f_mel_loss:
                        f_mel_loss.write(str(m_l)+"\n")
                    with open(os.path.join(self.log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                        f_mel_postnet_loss.write(str(m_p_l)+"\n")
                    with open(os.path.join(self.log_path, "duration_loss.txt"), "a") as f_d_loss:
                        f_d_loss.write(str(d_l)+"\n")
                    with open(os.path.join(self.log_path, "f0_loss.txt"), "a") as f_f_loss:
                        f_f_loss.write(str(f_l)+"\n")
                    with open(os.path.join(self.log_path, "energy_loss.txt"), "a") as f_e_loss:
                        f_e_loss.write(str(e_l)+"\n")"""
                    
                    # Backward
                    total_loss = total_loss / self.hp.acc_steps
                    total_loss.backward()
                    if current_step % self.hp.acc_steps != 0:
                        continue

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.hp.grad_clip_thresh)

                    # Update weights
                    self.scheduled_optim.step_and_update_lr()
                    self.scheduled_optim.zero_grad()
                    
                    # Print
                    if current_step % self.hp.log_step == 0:
                        Now = time.perf_counter()

                        str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                            epoch+1, self.hp.epochs, current_step, total_step)
                        str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                            t_l, m_l, m_p_l, d_l, f_l, e_l)
                        str3 = "Time Used: {:.3f}s({:.1f}min), Estimated Time Remaining: {:.3f}s.".format(
                            (Now-Start), ((Now-Start)/60), (total_step-current_step)*np.mean(Time))

                        print("\n" + str1)
                        print(str2)
                        print(str3)
                        
                        """with open(os.path.join(self.log_path, "log.txt"), "a") as f_log:
                            f_log.write(str1 + "\n")
                            f_log.write(str2 + "\n")
                            f_log.write(str3 + "\n")
                            f_log.write("\n")"""

                    """self.train_logger.add_scalar('Loss/total_loss', t_l, current_step)
                    self.train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                    self.train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                    self.train_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                    self.train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                    self.train_logger.add_scalar('Loss/energy_loss', e_l, current_step)"""
                    
                    if current_step % self.hp.save_step == 0:
                        os.makedirs(os.path.join(checkpoint_path, self.hp.user_id), exist_ok=True)
                        os.makedirs(os.path.join(checkpoint_path, self.hp.target_dir), exist_ok=True)

                        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(
                        )}, os.path.join(checkpoint_path, self.hp.target_dir, 'checkpoint_{}_{}.pth'.format(self.hp.dataset, current_step)))
                        print("save model at step {} ...".format(current_step))
                        
                        print(datetime.datetime.now() + datetime.timedelta(hours=9))

                        
                    end_time = time.perf_counter()
                    Time = np.append(Time, end_time - start_time)
                    if len(Time) == self.hp.clear_Time:
                        temp_value = np.mean(Time)
                        Time = np.delete(
                            Time, [i for i in range(len(Time))], axis=None)
                        Time = np.append(Time, temp_value)

if __name__ == "__main__":
    param = user_param.UserParam('hws0120', 'HW-man')
    hp = hparam(param)
    trainer = FS2Train(hp)

    trainer.train()
