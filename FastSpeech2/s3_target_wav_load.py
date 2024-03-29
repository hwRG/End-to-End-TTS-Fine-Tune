import boto3
import os 

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from param import user_param
from . import hparams


from pydub import AudioSegment
import scipy.io as sio
import re

from dotenv import load_dotenv
load_dotenv()


class S3TargetLoader:
    def __init__(self, hp):
        self.hp = hp
        self.target_path_origin = self.hp.direct_dir + '-origin'

        self.aws_access_id = os.environ.get("access_id")
        self.aws_secret_key = os.environ.get("secret_access_key")
        self.aws_bucket_name = os.environ.get("bucket_name")


    def load_from_s3(self):
        # set aws credentials 
        s3r = boto3.resource('s3', aws_access_key_id=self.aws_access_id,
            aws_secret_access_key=self.aws_secret_key)
        bucket = s3r.Bucket(self.aws_bucket_name)

        # mkdir
        os.makedirs(self.hp.base_dir, exist_ok=True)
        os.makedirs(self.hp.data_dir, exist_ok=True)
        os.makedirs(self.hp.direct_dir, exist_ok=True)
        os.makedirs(self.target_path_origin, exist_ok=True)

        for object in bucket.objects.filter(Prefix = self.hp.direct_dir):
            bucket.download_file(object.key, object.key)


    def m4a_to_wav(self):
        m4a_list = os.listdir(self.hp.direct_dir)
        m4a_list.sort()
        for m4a in m4a_list:
            track = AudioSegment.from_file(os.path.join(self.hp.direct_dir, m4a), format='m4a')

            wav = re.sub(r'[^0-9]', '', m4a[:-3])
            wav += '.wav'

            file_handle = track.export(os.path.join(self.target_path_origin, wav), format='wav')

        os.system('rm -rf {}'.format(self.hp.direct_dir))


    def down_sampling(self):
        os.makedirs(self.hp.direct_dir, exist_ok=True)

        wav_origin_list = os.listdir(self.target_path_origin)
        wav_origin_list.sort()
        for wav in wav_origin_list:
            sample_rate, _ = sio.wavfile.read(self.target_path_origin + '/' + wav)
            if sample_rate != self.hp.sampling_rate:
                os.makedirs(self.hp.direct_dir, exist_ok=True)
                os.system('ffmpeg -i {} -ac 1 -ar {} {} -y'.format(self.target_path_origin + '/' + wav, str(self.hp.sampling_rate), self.hp.direct_dir + '/' + wav))  
        
        os.system('rm -rf {}'.format(self.target_path_origin))


    def s3_target_load(self):
        self.load_from_s3()

        files = os.listdir(self.hp.direct_dir)
        if files[int(len(files)/2)][-3:] == 'm4a':
            self.m4a_to_wav()

        self.down_sampling()


if __name__ == '__main__':
    param = user_param.UserParam('hws0120', 'HW-man')
    hp = hparams.hparam(param)
    s3_loader = S3TargetLoader(hp)
    s3_loader.s3_target_load()