from jamo import h2j 
import json
import os, re
import unicodedata
import hparams as hp

class DataPreprocessing:
    def __init__(self):
        self.name = hp.dataset
        self.transcript = hp.meta_name
        self.dict_name = self.name + '_korean_dict.txt'

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

    def aligner(self):
        filters = '([.,!?])"'
        file_list = []

        with open('../../' + self.transcript, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                temp = line.split('|')
                
                file_dir, script = temp[0], temp[3]
                script = re.sub(re.compile(filters), '', script)
                # filters로 걸러지지 않는 항목 추가 제거
                #script = self.line_replace(script) 
                
                
                fn = file_dir[:-3] + 'lab'
                file_dir = os.path.join(fn)
                with open(file_dir, 'w', encoding='utf-8') as f:
                    f.write(script)

                file_list.append(os.path.join(file_dir))

        with open('../../' + self.transcript, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                temp = line.split('|')
                
                file_list.append(os.path.join(temp[0][:-3]) + 'lab')


        jamo_dict = {}
        for file_name in file_list:
            sentence =  open(file_name, 'r', encoding='utf-8').readline()
            jamo = h2j(sentence).split(' ')
            
            for i, s in enumerate(jamo):
                if s not in jamo_dict:
                    jamo_dict[s] = ' '.join(jamo[i])

        with open(self.dict_name, 'w', encoding='utf-8') as f:
            for key in jamo_dict.keys():
                content = '{}\t{}\n'.format(key, jamo_dict[key])
                f.write(content)
        print("Aligner Done\n")


    def mfa_train(self):
        print("MFA Training Start.. \n")

        os.system('mfa train_g2p ' + self.dict_name + ' ' + self.name + '_korean.zip --clear')
        print("MFA train_g2p Done\n")

        os.system('mfa g2p ' + self.name + '_korean.zip . ' + self.name + '_korean.txt')
        print("MFA g2p Done\n")
    
        os.system('mfa train . ' + self.name + '_korean.txt ./textgrids --clean')
        
        os.system('mv ~/Documents/MFA/wavs_train_acoustic_model/sat_2_ali/textgrids ./')
        #os.system('zip -r {}_textgrids.zip textgrids'.format(hp.dataset))
        #os.system('mv {}_textgrids.zip '.format(hp.dataset) + first_dir) # 압축 후 최상위 디렉토리에 zip 파일로 생성
        print("MFA Training Done! \n")
        

    def lab_separate(self):
        lab_list = os.listdir('.')
        for lab in lab_list:
            if lab[-3:] == 'lab':
                os.system('rm ' + lab)

    # RUN
    def data_preprocess(self):
        # 0) 데이터 경로로 이동
        os.chdir('../{}'.format(hp.data_path))
        # API로 활용 시
        # os.system('cd End-to-End-TTS-Fine-Tune/{}'.format(hp.data_path))

        # 1) mfa를 위해 데이터마다 lab 파일 생성
        self.aligner()

        # 2) 순차적으로 mfa 수행하여 textgrids 생성 및 압축  
        self.mfa_train()

        # 3) 디렉토리에 lab 파일 제거
        self.lab_separate()

        # 4) 기존 경로로 이동
        os.system('cd -')

if __name__ == "__main__":
    processor = DataPreprocessing()  
    processor.data_preprocess()