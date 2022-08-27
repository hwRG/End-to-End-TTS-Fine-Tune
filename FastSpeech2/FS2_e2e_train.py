
import os
import subprocess
from . import s3_target_wav_load, data_preprocessing,  preprocess, train

class FS2Train:
    def __init__(self, hp):
        self.hp = hp

    def E2E_FS2_train(self):
        s3_loader = s3_target_wav_load.S3TargetLoader(self.hp)
        s3_loader.s3_target_load()
        print('S3 Complete!')
        
        Processor = data_preprocessing.DataPreprocessing(self.hp)
        Processor.data_preprocess()
        
        preprocessor = preprocess.Preprocess(self.hp)
        preprocessor.preprocess()

        FS2_trainer = train.FS2Train(self.hp)
        FS2_trainer.train()

        