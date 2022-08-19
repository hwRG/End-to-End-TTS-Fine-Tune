from dto.TTS_dto import TTSTrainDto, TTSInferenceDto

from FastSpeech2 import train as FS2train
from FastSpeech2 import hparams, data_preprocessing
from FastSpeech2 import s3_target_wav_load
from HiFiGAN import train as HGtrain
from param import user_param

from FastSpeech2 import synthesize

import os
import uvicorn

from fastapi import FastAPI
app = FastAPI()

@app.get("/tts/train")
async def tts_train(target_inf: TTSTrainDto):
    # param을 설정하고 hparam에 통째로 넘겨줌 그리고 그 hp를 함수에서 계속활용
    param = user_param.UserParam(target_inf.caregiver_id, target_inf.voice_target)
    hp = hparams.hparam(param)

    s3_loader = s3_target_wav_load.S3TargetLoader(hp)
    s3_loader.s3_target_load()

    # 가상 환경에서 aligner 수행
    os.system('. /opt/conda/etc/profile.d/conda.sh')
    os.system('conda activate aligner')
    Processor = data_preprocessing.DataPreprocessing(hp)
    Processor.data_preprocess()
    os.system('conda activate base')

    FS2_trainer = FS2train.FS2Train(hp)
    FS2_trainer.train()

    HG_trainer = HGtrain.HiFiGANTrain(param)
    HG_trainer.train()

    return {"response": "Fine-tune train Complete!"} 


@app.get("/tts/inference")
async def tts_inference(target_text: TTSInferenceDto):
    param = user_param.UserParam(target_text.caregiver_id, target_text.voice_target)
    hp = hparams.hparam(param)
    
    synthesizer = synthesize.Synthesizer(hp)
    wav_path = synthesizer.synthesize(target_text.msg)

    # wav 파일 경로를 받아 backend로 전달하면 될 듯
    return wav_path

    

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
        host="0.0.0.0",
        reload=True
    )