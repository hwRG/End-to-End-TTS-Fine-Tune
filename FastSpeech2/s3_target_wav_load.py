import boto3
import os 

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from param import param

from pydub import AudioSegment
import scipy.io as sio
import re

target_path_origin = param.direct_dir + '-origin'

def load_from_s3():
    # set aws credentials 
    s3r = boto3.resource('s3', aws_access_key_id='AKIA6AIFPD7P4RRW2TUT',
        aws_secret_access_key='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')
    bucket = s3r.Bucket('nearby-development-bucket')

    # mkdir
    if not os.path.exists(param.base_dir):
        os.mkdir(param.base_dir)
    if not os.path.exists(param.data_dir):
        os.mkdir(param.data_dir)
    if not os.path.exists(param.direct_dir):
        os.mkdir(param.direct_dir)
    if not os.path.exists(target_path_origin):
        os.mkdir(target_path_origin)

    for object in bucket.objects.filter(Prefix = param.direct_dir):
        bucket.download_file(object.key, object.key)


def m4a_to_wav():
    m4a_list = os.listdir(param.direct_dir)
    m4a_list.sort()
    for m4a in m4a_list:
        track = AudioSegment.from_file(os.path.join(param.direct_dir, m4a), format='m4a')

        wav = re.sub(r'[^0-9]', '', m4a[:-3])
        wav += '.wav'

        file_handle = track.export(os.path.join(target_path_origin, wav), format='wav')

    os.system('rm -rf {}'.format(param.direct_dir))

def down_sampling():
    if not os.path.exists(param.direct_dir):
        os.mkdir(param.direct_dir)

    wav_origin_list = os.listdir(target_path_origin)
    wav_origin_list.sort()
    for wav in wav_origin_list:
        sample_rate, _ = sio.wavfile.read(target_path_origin + '/' + wav)
        if sample_rate != 22050:
            if not os.path.exists(param.direct_dir):
                os.mkdir(param.direct_dir)
            os.system('ffmpeg -i {} -ac 1 -ar 22050 {} -y'.format(target_path_origin + '/' + wav, param.direct_dir + '/' + wav))  
    
    os.system('rm -rf {}'.format(target_path_origin))

if __name__ == '__main__':
    os.chdir('..')
    load_from_s3()
    files = os.listdir(param.direct_dir)
    if files[int(len(files)/2)][-3:] == 'm4a':
        m4a_to_wav()
    down_sampling()