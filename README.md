# End-to-End TTS Fine-Tune

## Introduction

- 본 프로젝트는 ‘**친근한 목소리를 가진 노인 맞춤 인공지능 비서**'의 목소리 제작을 목표로 합니다. 더이상 '시리', '빅스비', '아리'의 목소리에 의지하는 것이 아닌, 단 7분의 녹음본으로 가족, 친구 목소리로 인공지능 비서 목소리를 대체합니다.
- AI 스피커라는 즉각 생성에 대응하고자 **Non-Autoregressive Acoustic Model FastSpeech2과 GAN 기반 Vocoder Model HiFi-GAN**을 채택하여 퀄리티와 생성 속도를 고려했습니다.
- 성능 향상을 위해 모델에 추가적인 커스텀을 진행한 [D-vector Multi Speaker FastSpeech2](https://github.com/hwRG/FastSpeech2-Pytorch-Korean-Multi-Speaker)와 [HiFi-GAN](https://github.com/hwRG/HiFi-GAN-Pytorch) 모델을 Fine-tuning 환경을 구축하여, shell script로 쉽게 학습과 합성을 수행할 수 있도록 구성했습니다.
- 또한 앱에 실시간 TTS를 제공하기 위해 FastAPI를 활용하여 서버를 구성하고 Backend와 연동합니다.

<br>

## Project Purpose

1. Acoustic-Custom Fastspeech2, Vocoder-HiFiGAN 모델을 활용한 고성능 및 고속 음성 합성

2. 소량의 데이터로 개인화를 위한 Transfer Learning을 활용해 합리적인 성능 제공  
3. 한국어 데이터셋에 Fine-tuning 과정이 실시간으로 이루어지도록 API 제공

<br>

## Dataset

- dataset 폴더에 속한 fine_tune_transcript.txt에 따라 스마트폰으로 숫자에 따라 100문장을 녹음하고, m4a 파일을 sampling rate가 16000인 wav 파일로 변환합니다.  (ffmpeg)

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
![](https://velog.velcdn.com/images/hws0120/post/0e5f6fbd-002a-4f0f-a172-b0201480bb23/image.png)
학습과 합성이 수행되는 파이프라인입니다.<br>

학습 / 합성 과정에서 각각의 컨테이너를 구축하고 그림과 같은 과정을 수행합니다.<br>

잘 학습된 Multi-Speaker HiFi-GAN checkpoint가 있다면 HiFi-GAN 학습은 생략해도 됩니다.
