FROM nvidia/cuda:10.2-runtime-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get install -y wget libsndfile1 ffmpeg nano build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN pip install -v boto3 nltk python-mecab-ko tweepy==3.10.0 numpy unidecode jamo \
    matplotlib librosa tqdm inflect torch==1.8.1 pandas \
    tgt pyworld torchaudio==0.8.1 tensorboard==2.8.0 pytorch-ignite \ 
    termcolor visdom ftfy seaborn timm==0.4.5 pydub scipy multiprocess \ 
    webrtcvad umap-learn==0.5.2 protobuf==3.20.0

CMD ["sh", "run_train.sh"]