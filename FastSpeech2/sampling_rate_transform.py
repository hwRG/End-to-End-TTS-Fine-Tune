import os
from data import data_processing
import hparams as hp
import scipy.io as sio


def main():
    in_dir = hp.data_path

    speakers = os.listdir(os.path.join(in_dir, 'wavs'))
    sample_data = os.listdir(os.path.join(in_dir, 'wavs', speakers[1]))
    for sample_wav in sample_data:
        if sample_wav[-3:] == 'wav':
            temp = sample_wav
            print(temp, '\n\n\n')
            break
    sample_rate, _ = sio.wavfile.read(os.path.join(in_dir, 'wavs', speakers[1], temp))

    if sample_rate != hp.sampling_rate:
        os.system('mv ' + in_dir + '/wavs ' + in_dir + '/wavs_{}'.format(str(sample_rate)))

        for speaker in speakers:
            dir_list = os.listdir(os.path.join(in_dir, 'wavs_{}'.format(str(sample_rate)), speaker))
            wav_list = []
            for dir in dir_list:
                if '.wav' in dir:
                    wav_list.append(dir)

            for wav in wav_list:
                wav_path = os.path.join(in_dir, 'wavs', speaker, wav)
                wav_before_path = os.path.join(in_dir, 'wavs_{}'.format(str(sample_rate)), speaker, wav)

                os.makedirs(os.path.join(in_dir, 'wavs'), exist_ok=True)
                os.makedirs(os.path.join(in_dir, 'wavs', speaker), exist_ok=True)

                os.system("ffmpeg -i {} -ac 1 -ar {} {}".format(wav_before_path, str(hp.sampling_rate), wav_path))
    
if __name__ == "__main__":
    main()
