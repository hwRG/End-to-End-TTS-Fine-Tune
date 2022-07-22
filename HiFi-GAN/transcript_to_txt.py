import random
import os
import glob

transcript = glob.glob('dataset/*.txt')

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

def make_txt():
    os.chdir('dataset')
    transcript = glob.glob('*.txt')
    transcript = transcript[0]

    train_val = []
    with open(transcript, 'r', encoding='utf8') as file:    # hello.txt 파일을 읽기 모드(r)로 열기
        line = None    # 변수 line을 None으로 초기화
        while line != '':
            line = file.readline()
            if line == '':
                break
            line = line.replace('.wav', '')
            name = line[:line.find('|')]
            line = line[line.find('|')+1:]
            text = line[:line.find('|')]
            text = line_replace(text)
            train_val.append(name + '|' + text + '|' + text + '\n')
    print('Sentence Count:', len(train_val))


    train, val = list(), list()

    for line in train_val:
        rand = random.randrange(0,7)
        if rand < 1:
            val.append(line)
        else:
            train.append(line)

    f = open('training.txt', 'w')
    for line in train:
        f.write(line)
    f.close()

    f = open('validation.txt', 'w')
    for line in val:
        f.write(line)
    f.close()
    os.chdir('../')


if __name__ == '__main__':
    make_txt()