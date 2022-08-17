import boto3
import os 

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from param import param

from pydub import AudioSegment
import scipy.io as sio
import re

import hparams as hp

from dotenv import load_dotenv
load_dotenv()


class S3TargetLoader:
    def __init__(self):
        self.target_path_origin = param.direct_dir + '-origin'

        self.aws_access_id = os.environ.get("access_id")
        self.aws_secret_key = os.environ.get("secret_access_key")
        self.aws_bucket_name = os.environ.get("bucket_name")

    def load_from_s3(self):
        # set aws credentials 
        s3r = boto3.resource('s3', aws_access_key_id=self.aws_access_id,
            aws_secret_access_key=self.aws_secret_key)
        bucket = s3r.Bucket(self.aws_bucket_name)

        # mkdir
        if not os.path.exists(param.base_dir):
            os.mkdir(param.base_dir)
        if not os.path.exists(param.data_dir):
            os.mkdir(param.data_dir)
        if not os.path.exists(param.direct_dir):
            os.mkdir(param.direct_dir)
        if not os.path.exists(self.target_path_origin):
            os.mkdir(self.target_path_origin)

        for object in bucket.objects.filter(Prefix = param.direct_dir):
            bucket.download_file(object.key, object.key)


    def m4a_to_wav(self):
        m4a_list = os.listdir(param.direct_dir)
        m4a_list.sort()
        for m4a in m4a_list:
            track = AudioSegment.from_file(os.path.join(param.direct_dir, m4a), format='m4a')

            wav = re.sub(r'[^0-9]', '', m4a[:-3])
            wav += '.wav'

            file_handle = track.export(os.path.join(self.target_path_origin, wav), format='wav')

        os.system('rm -rf {}'.format(param.direct_dir))

    def down_sampling(self):
        if not os.path.exists(param.direct_dir):
            os.mkdir(param.direct_dir)

        wav_origin_list = os.listdir(self.target_path_origin)
        wav_origin_list.sort()
        for wav in wav_origin_list:
            sample_rate, _ = sio.wavfile.read(self.target_path_origin + '/' + wav)
            if sample_rate != hp.sampling_rate:
                if not os.path.exists(param.direct_dir):
                    os.mkdir(param.direct_dir)
                os.system('ffmpeg -i {} -ac 1 -ar {} {} -y'.format(self.target_path_origin + '/' + wav, str(hp.sampling_rate), param.direct_dir + '/' + wav))  
        
        os.system('rm -rf {}'.format(self.target_path_origin))

    def s3_target_load(self):
        # 최상위 디렉토리에서 실행할 경우 chdir 주석처리
        os.chdir('..')

        self.load_from_s3()

        files = os.listdir(param.direct_dir)
        if files[int(len(files)/2)][-3:] == 'm4a':
            self.m4a_to_wav()

        self.down_sampling()

if __name__ == '__main__':
    s3_loader = S3TargetLoader()
    s3_loader.s3_target_load()