# transcript에 g2pk를 사용할 경우, g2pk 폴더의 g2pk.py의 7과 8의 주석 해제

### Data Preprocessing
## 1. Json to Transcript
## 2. Aligner
## 3. Text Replace

from jamo import h2j 
import json
import os, re, tqdm
import unicodedata
from tqdm import tqdm
import hparams as hp

from g2pk.g2pk import G2p
from jamo import h2j

name = hp.dataset

first_dir = os.getcwd()

transcript = name + '_transcript.txt'
dict_name = name + '_korean_dict.txt'

data_dir = 'wavs'
json_label_dir = 'label'


def applied_g2pk(text):
    g2p=G2p()
    phone = g2p(text) 
    phone = h2j(phone)
    print('After g2p: ',phone)
    print()
    return phone

def main():
    processed_line = []
    with open(transcript, 'r', encoding='utf-8') as f:
        ckpt_list = [-1]
        #for ckpt in ckpts:
        #    ckpt_list.append(int(re.sub(r'[^0-9]', '', ckpt)))
        ckpt_list.sort()
        print(ckpt_list)

        for i, line in enumerate(f.readlines()):
            if i+1 <= int(ckpt_list[-1]):
                continue

            if i % 50000 == 0 and i != 0:
                with open('transcript_ckpt/processed_' + str(i) + '_' + transcript, 'w', encoding='utf-8') as f:
                    for write_line in processed_line:
                        f.write(write_line)
                if processed_line:
                    return
                print('Checkpoint', str(i+1))
                
            temp = line.split('|')
            file_dir, sec, sec2, script, fourth, fifth = temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]
            print('Before g2p: ',script, i+1)
            script = applied_g2pk(script)
            processed_line.append(str(file_dir) + '/' + str(sec) + '|' + str(sec2) + '|' + str(script) + '|' +  str(fourth) + '|' + str(fifth))

    
    with open('processed_' + transcript, 'w', encoding='utf-8') as f:
        for line in processed_line:
            f.write(line)

if __name__ == "__main__":
    os.chdir('dataset/' + hp.dataset)
    main()