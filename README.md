# End-to-End TTS Fine-Tune

FastSpeech2와 HiFi-GAN을 활용해 End-to-End 한국어 음성합성을 쉽게 수행합니다.



## Introduction

- 본 프로젝트는 **‘보이는 개인화 AI 스피커’ 프로젝트의 TTS 자동화**를 목표합니다. 기존에 식상한 '시리', '빅스비', '아리'의 목소리가 아닌, 단 6분의 데이터로 주변 사람의 목소리로 대체합니다. (ex. 배우자, 아들, 딸, 부모님 등)

- AI 스피커라는 즉각적인 생성에 대응하고자 **Non-Autoregressive Acoustic Model FastSpeech2**과 **GAN 기반 Vocoder Model HiFi-GAN**을 채택하여 퀄리티와 생성 속도 모두 고려합니다.
- 미리 구현된 [FastSpeech2](https://github.com/hwRG/FastSpeech2-Pytorch-Korean-Multi-Speaker)와 [HiFi-GAN](https://github.com/hwRG/HiFi-GAN-Pytorch)을 Fine-tuning 환경에 맞게 수정하였으며, shell script로 간편히 데이터별로 학습과 합성을 수행할 수 있습니다.

<br>

## Project Purpose

1. Acoustic-Fastspeech2, Vocoder-HiFiGAN 모델을 활용해 빠른 생성과 고성능 합성

2. 소량의 데이터로 개인화하기 위해 Transfer Learning을 활용해 합리적인 성능 제공  
3. 한국어 데이터셋에 fine-tune 과정이 end-to-end로 이루어지도록 shell script 제공

<br>

## Dataset

- dataset 폴더에 속한 fine_tune_transcript.txt에 따라 스마트폰으로 숫자에 따라 100문장을 녹음하고, m4a 파일을 sampling rate가 16000인 wav 파일로 변환한다.  

![](https://velog.velcdn.com/images/hws0120/post/ffa4b385-9a4c-49a4-a01b-3ccb45e52a90/image.png) ![](https://velog.velcdn.com/images/hws0120/post/7efaba35-5558-469a-8c65-65dbe696d195/image.png)
- 그림과 같이 변환된 wav 파일 100개를 본인 이니셜 폴더에 넣어 준비합니다.

<br>

## Contribution
1. Fine-tune에 맞게 모델 코드 수정

   + FastSpeech2와 HiFi-GAN 수정 및 통합
   + Dataset, ckpt, results 디렉토리를 최상위 디렉토리에 데이터셋 별로 구분
2. Shell Script를 통한 간편한 preprocess, train, synthesis 수행 
   - dataset 디렉토리의 바꿔줌으로써 
3. 독자적인 docker image 제공
   - 복잡한 추가 의존성 패키지 추가 없이 바로 수행 가능한 이미지 제공 
   - [도커 허브 링크](https://hub.docker.com/r/hws0120/e2e_speech_synthesis)를 통해 latest 이미지 불러오기

<br>



## Previous Works
- [링크]()에서 FastSpeech2와 HiFi-GAN에 대한 pre-trained ckpt를 불러와 모델 각각 ckpt 디렉토리에 보관합니다. (FastSpeech2 25만 step, HiFi-GAN 30만 step) 

- 학습과 합성을 위해 모든 의존성 패키지가 포함된 도커 이미지를 불러와 실행합니다.

  ```
  docker pull hws0120/e2e_speech_synthesis 
  ```

- run_FS2_preprocessing.sh 단계는 conda 명령어로 도커 내 가상환경에 접속하고 python 패키지 jamo를 설치합니다.

  ```
  conda activate aligner
  pip install jamo
  ```

- run_FS2_train 또는 synthesis를 수행하기 위해 가상 환경을 종료합니다.

  ```
  conda activate base
  ```

<br>

## Preprocessing

- 위 항목을 모두 충족하면 shell script를 실행하여 mfa를 추출합니다.

  ```
  sh run_FS2_preprocessing.sh
  # Enter the dataset name
  [Dataset_Name](ex. HW)
  ```

  <br>

## Train

- 성공적으로 textgrid를 생성하면 가상환경을 종료하고 학습 스크립트를 실행합니다.  

  ```
  sh run_FS2_train.sh
  # Enter the dataset name
  [Dataset_Name](ex. HW)
  ```

- FastSpeech2를 5000 step 학습해 끝나면 HiFi-GAN도 스크립트를 실행합니다.  

  ```
  sh run_HiFi-GAN_train.sh
  # Enter the dataset name
  [Dataset_Name](ex. HW)
  ```

  <br>

## Synthesize

- ckpt 폴더에 학습된 모델이 준비되면 합성을 위한 스크립트를 실행합니다.

  ```
  sh run_FS2_synthesize.sh
  # Enter the dataset name
  [Dataset_Name](ex. HW)
  ```

<br>

## Project Pipeline

학습과 합성이 수행되는 파이프라인입니다.<br>

구성에 따라 3개로 나뉘어질 컨테이너를 하나의 이미지에 담았습니다. 

1. Transcript 생성, 파일명 간소화, MFA로 Textgrid 추출 
2. FastSpeech2 모델에 활용할 processed 데이터 추출 및 학습 
3. HiFi-GAN 모델 학습