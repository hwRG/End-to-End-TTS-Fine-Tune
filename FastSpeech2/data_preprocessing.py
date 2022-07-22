### Data Preprocessing
## 1. Json to Transcript
## 2. Aligner
## 3. Text Replace

from jamo import h2j 
import json
import os, re
import unicodedata
import hparams as hp

name = hp.dataset

first_dir = os.getcwd()

transcript = hp.meta_name
dict_name = name + '_korean_dict.txt'

data_dir = 'wavs'
json_label_dir = 'label'


def change_name(base_dir, format):
    print('Change', format, 'name')
    cnt = 0
    speaker_table = os.listdir(base_dir)
    new_speaker_table = []
    
    for speaker in speaker_table:
        if cnt == 0:
            os.chdir(base_dir)
            
        new_speaker_name = re.sub(r'[^0-9]', '', speaker)
        
        overlap = 1
        while new_speaker_name in new_speaker_table:
            print(new_speaker_name, 'is dangerous')
            new_speaker_name = str(overlap) + new_speaker_name[1:]
            overlap += 1
        
        new_speaker_table.append(re.sub(r'[^0-9]', '', new_speaker_name))
        print(new_speaker_name, 'ok')
        
        temp = 0
        for wav in os.listdir(speaker):
            if temp == 0:
                os.chdir(speaker)
            new_wav_name = re.sub(r'[^0-9]', '', wav)

            # new wav_name을 그대로 사용해야 함
            if new_wav_name[:len(new_speaker_name)] != wav:
                if new_wav_name[:len(new_speaker_name)] == new_speaker_name:
                    new_wav_name = new_wav_name + wav[-(len(format)+1):]
                else:
                    new_wav_name = new_speaker_name + new_wav_name + wav[-(len(format)+1):]
                os.rename(wav, new_wav_name)
            
            temp+=1; cnt +=1
            
        os.chdir('../')
        os.rename(speaker, new_speaker_name)
    print(cnt,'All Done', end='\n\n')
    os.chdir('../')


def json_to_transcripts():
    speakers = os.listdir(json_label_dir)
    speakers.sort()
    print(len(speakers), "speaker's are Sorted.")
    os.chdir(json_label_dir)

    utterance_text = []
    cnt = 1
    for speaker in speakers:
        for file in os.listdir(speaker):
            if cnt % 1000 == 0:
                print(cnt, 'Done')

            utterance_set = []
            with open(os.path.join(speaker, file)) as f:
                json_data = json.load(f)
                utterance_set.append(file[:-4] + 'wav')
                utterance_set.append(line_replace(json_data['발화정보']['stt']))
                
                # unicodedata.normalize로 글자의 위치까지 기록해 나열
                sep_text = unicodedata.normalize('NFD',line_replace(json_data['발화정보']['stt']))
                utterance_set.append(sep_text)
                
                utterance_set.append(round(float(json_data['발화정보']['recrdTime']),1))
                
                utterance_text.append(utterance_set)
            cnt+=1

    print(cnt-1, 'All Done')
    os.chdir('../')
    with open(transcript, "w") as file:
        for utt in utterance_text:
            file.write(utt[0][:6] + '/' + utt[0] + '|' + utt[1] + '|' + utt[1] + '|' +  utt[2] + '|' +  str(utt[3]) + '|' +  'None\n')


def line_replace(line):
    line = line.replace('(SP:)', '')
    line = line.replace('(SP:', '')
    line = line.replace('(SN:)', '')
    line = line.replace('(SN:', '')
    line = line.replace('(NO:)', '')
    line = line.replace('(NO:', '')
    line = line.replace('spn', '')
    line = line.replace('', '')
    line = line.replace('', '')
    line = line.replace('', '')
    line = line.replace('', '')
    line = line.replace('毛', '')
    line = line.replace(')', '')
    line = line.replace('(', '')
    line = line.replace('"', '')
    line = line.replace('.', '')
    line = line.replace('[', '')
    line = line.replace(',', '')
    line = line.replace('!', '')
    line = line.replace('?', '')
    line = line.replace(']', '')
    line = line.replace('.', '')
    line = line.replace('  ', ' ')
    return line

def aligner():
    filters = '([.,!?])"'
    file_list = []

    with open('../' + transcript, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split('|')
            
            file_dir, script = temp[0], temp[3]
            script = re.sub(re.compile(filters), '', script)

            # filters로 걸러지지 않는 항목 추가 제거
            script = line_replace(script) 
            
            
            fn = file_dir[:-3] + 'lab'
            file_dir = os.path.join(data_dir, fn)
            with open(file_dir, 'w', encoding='utf-8') as f:
                f.write(script)

            file_list.append(os.path.join(file_dir))

    with open('../' + transcript, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split('|')
            
            file_list.append(os.path.join(data_dir, temp[0][:-3]) + 'lab')


    jamo_dict = {}
    for file_name in file_list:
        sentence =  open(file_name, 'r', encoding='utf-8').readline()
        jamo = h2j(sentence).split(' ')
        
        for i, s in enumerate(jamo):
            if s not in jamo_dict:
                jamo_dict[s] = ' '.join(jamo[i])

    with open(dict_name, 'w', encoding='utf-8') as f:
        for key in jamo_dict.keys():
            content = '{}\t{}\n'.format(key, jamo_dict[key])
            f.write(content)
    print("Aligner Done\n")


def mfa_train():
    print("MFA Training Start.. \n")

    os.system('mfa train_g2p ' + dict_name + ' ' + name + '_korean.zip --clear')
    print("MFA train_g2p Done\n")

    os.system('mfa g2p ' + name + '_korean.zip ' + data_dir + ' ' + name + '_korean.txt')
    print("MFA g2p Done\n")
   
    os.system('mfa train ' + data_dir + ' ' + name + '_korean.txt ./textgrids --clean')
    
    os.system('mv ~/Documents/MFA/wavs_train_acoustic_model/sat_2_ali/textgrids ./')
    #os.system('zip -r {}_textgrids.zip textgrids'.format(hp.dataset))
    #os.system('mv {}_textgrids.zip '.format(hp.dataset) + first_dir) # 압축 후 최상위 디렉토리에 zip 파일로 생성
    print("MFA Training Done! \n")
    

def lab_separate():
    lab_list = os.listdir('wavs')
    for lab in lab_list:
        if lab[-3:] == 'lab':
            os.system('rm ' 'wavs/' + lab)


if __name__ == '__main__':
    os.chdir('../dataset/_' + hp.dataset)

    # 디렉토리와 파일에 숫자 제외 모두 제거, 중복되지 않도록 조정
    #change_name('wavs', 'wav')
    #change_name('label', 'json')

    # AIHub 데이터 기준 json 데이터를 transcript로 변환
    #json_to_transcripts()

    # 1) mfa를 위해 데이터마다 lab 파일 생성
    aligner()

    # 2) 순차적으로 mfa 수행하여 textgrids 생성 및 압축  
    mfa_train()

    # 3) 디렉토리에 lab 파일 제거
    lab_separate()