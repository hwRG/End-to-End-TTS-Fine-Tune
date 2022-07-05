# FastSpeech2-Pytorch-Korean-Multi-Speaker

FastSpeech2에 HiFi-GAN Vocoder를 결합하여, 한국어 Multi-Speaker TTS로 구현한 프로젝트 입니다. 



## Introduction

- 본 프로젝트는 **‘보이는 개인화 AI 스피커’ 프로젝트의 TTS 개발**을 목표합니다. 기존에 식상한 '시리', '빅스비', '아리'의 목소리가 아닌 사용자가 원하는 주변 사람의 목소리로 대체합니다. (ex. 배우자, 아들, 딸, 부모님 등)

- AI 스피커라는 즉각적인 생성에 대응하기 위해 기존에 뛰어난 성능의 Tacotron2와 Waveglow 대신 **Non-Autoregressive Acoustic Model FastSpeech2**과 **GAN 기반 Vocoder Model HiFi-GAN**을 채택하여 퀄리티와 생성 속도 모두 고려합니다.

- [DLLAB](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)에서 구현한 한국어 데이터셋 [KSS](http://kaggle.com/bryanpark/korean-single-speaker-speech-dataset)에 대응하는 FastSpeech2 소스 코드를 기반으로 구현되었습니다.

<br>

## Project Purpose

1. 빠른 합성 속도와 높은 성능을 위해 Acoustic-Fastspeech2, Vocoder-HiFiGAN 모델 활용
2. 소량의 데이터로 개인화하기 위해 Transfer Learning 활용 (~~+ Zero-shot Cloning~~ **Side Project**)
3. Pre-train을 위한 Multi-Speaker를 학습하기 위해 Speaker Embedding 구현 
4. 한국어 데이터셋에 학습 과정이 end-to-end로 수행되도록 파이프라인 구성

<br>

## Dataset

- Pre-train을 위해 AIHub의 [자유대화 음성](https://aihub.or.kr/aidata/30703)을 활용해 학습을 수행합니다. 
  - 평균 1시간 30분, 퀄리티를 고려하여 남성 30명과 여성 28명 데이터로 학습 
  - 각 화자마다 고유한 숫자 ID를 전처리 과정에서 부여
- Fine-tune을 위해 KSS 스크립트 일부를 참고하여, 100문장-300문장-600문장 단위로 새로운 화자의 목소리를 녹음하여 성능을 평가합니다.

<br>

## Contribution (Add from Previous Project)

활용한 코드에서 추가된 내용은 다음과 같습니다.

1. **Speaker Embedding 구현**  (Korean Multi-Speaker FastSpeech2)

   + 모델에 Embedding layer를 추가
   + Encoder output과 더하는 코드 구현 (Embedding, Speaker Integrator)
   + Embedding 정보를 가져오고 저장하는 get_speakers() 함수 구현

2. data_preprocessing.py - 아래 항목을 모두 포함하는 end-to-end 데이터 전처리 구현

   ![data_preprocessing](/asset/data_preprocessing.png)

3. 긴 문장에 대한 불안정한 합성 시 대응

   - 특수문자 단위로(문장 단위) 끊어 합성 후 이어 붙이도록 설정
   
   ![cut_and_synthesize](/asset/cut_and_synthesize.png)
   
4. G2pk 소스 코드를 불러와 숫자, 영어만 변환하도록 적용

   - 기존 [G2pk](https://github.com/Kyubyong/g2pK)의 패키지를 pip 설치 없이 숫자, 영어만 한글로 변환 하도록 수정

<br>



## Previous Works

![](https://velog.velcdn.com/images/hws0120/post/87b67f84-943a-491b-8729-bae16c6ec267/image.png) ![](https://velog.velcdn.com/images/hws0120/post/6fa91612-305c-495d-8901-6ee59d8d690e/image.png)

- 그림과 같이 wav 디렉토리와 json 또는 transcript 파일을 dataset/데이터명 디렉토리에 저장합니다.

- Kaldi로 구현된 [Montral Forced Alinger](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html)를 오디오 데이터에 학습하여 **Textgrid**를 얻습니다.
  ```
  # lab 생성, mfa 학습, lab 분리
  python data_preprocessing.py 
  ```
- 학습 중 평가를 위해 [HiFi-GAN](https://github.com/hwRG/HiFi-GAN-Pytorch)으로 학습한 generator를 vocoder/pretrained_models 디렉토리에 저장합니다.

1. 데이터를 형식에 따라 transcript를 직접 작성하거나, data_preprocessing.py의 함수를 참고하여 transcrip 생성
2. 생성된 transcript와 데이터의 디렉토리를 dataset에 보관 후 data_preprocessing.py를 실행
3. MFA 작업이 완료되고 Textgrid.zip 파일이 최상위 디렉토리에 생성된 것을 확인
4. preprocess.py를 수행하고 preprocessed 폴더에 생성된 전처리 데이터 확인


<br>

## Train

- hparam.py의 batch size, HiFi-GAN generator의 path를 설정하고 학습을 시작합니다.

  ```
  python train.py
  ```

- 재학습을 하게 될 경우, restore_step을 추가하여 재학습이 가능합니다.
  ```
  python train.py --restore_step [step]
  ```



## Transfer Learning

1. Multi-Speaker에 대한 Pre-train을 수행할 경우, Pre-train 학습 시 자동으로 생성된 speaker_info.json 저장

2. 디렉토리 최상단에 speaker_info.json을 넣고 ckpt/데이터이름 디렉토리에 pth.tar 체크포인트 복사

3. Train에서 재학습을 수행하는 것과 동일하게 파이썬 실행

   ```
   python train.py --restore_step [pre-train의 step]
   ```

<br>

## Synthesize

- Snythesize.py 파일로 합성합니다.
  ```
  python synthesize.py --step [step수]
  ```
  - 임의로 제시한 대본으로 합성 1, 2, 3 선택
  - 직접 작성한 대본으로 합성할 경우 4 선택

<br>

## Model Pipeline

본 파이프라인은 서비스에 해당되는 TTS 학습 및 생성에 대한 flow 파이프라인 입니다.

![Transfer_Learning_Pipeline](/asset/Transfer_Learning_Pipeline.png)

- 컨테이너는 크게 4개로 분류됩니다. 
  1. 데이터의 path와 유저 정보 등을 담고 있는 데이터베이스 컨테이너
  2. Transcript 생성, 파일명 간소화, MFA로 Textgrid 추출, 모델에 필요한 Data Preprocessing 컨테이너
  3. Pre-Training을 위한 학습 컨테이너
  4. 새로운 데이터에 Fine-Tuning을 위한 학습 컨테이너

- 실제 서비스 상황엔 Pre-trianing 컨테이너 외 3개 컨테이너만 작동하게 됩니다.