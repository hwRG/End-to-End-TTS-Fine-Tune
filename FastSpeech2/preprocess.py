import os
from .data import data_processing
import scipy.io as sio

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from param import user_param
from . import hparams

class Preprocess():
    def __init__(self, hp):
        self.hp = hp

    def write_metadata(self, train, val, out_dir):
        with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            for m in train:
                f.write(m + '\n')
        with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
            for m in val:
                f.write(m + '\n')

    def preprocess(self):
        in_dir = self.hp.direct_dir
        out_dir = self.hp.preprocessed_path
        meta = self.hp.meta_name

        # processed 데이터 경로 지정
        mel_out_dir = os.path.join(out_dir, "mel")
        os.makedirs(mel_out_dir, exist_ok=True)

        ali_out_dir = os.path.join(out_dir, "alignment")
        os.makedirs(ali_out_dir, exist_ok=True)

        f0_out_dir = os.path.join(out_dir, "f0")
        os.makedirs(f0_out_dir, exist_ok=True)

        energy_out_dir = os.path.join(out_dir, "energy")
        os.makedirs(energy_out_dir, exist_ok=True)
        
        os.makedirs('./preprocessed/{}'.format(self.hp.target_dir), exist_ok=True)

        # Textgrids 그대로 옮기기
        os.system('cp -r {}/textgrids {}/textgrids'.format(self.hp.direct_dir, self.hp.preprocessed_path))
        
        # train, val은 리스트로, 파일 위치와 텍스트를 받아 저장
        train, val = data_processing.build_from_path(in_dir, out_dir, meta, self.hp)

        if len(train) != 0:
            self.write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    param = user_param.UserParam('hws0120', 'HW-man')
    hp = hparams.hparam(param)
    preprocessor = Preprocess(hp)
    preprocessor.preprocess()
