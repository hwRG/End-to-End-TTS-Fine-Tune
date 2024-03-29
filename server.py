from dto import TTS_Dto

from FastSpeech2 import hparams, FS2_e2e_train
from HiFiGAN import train as HGtrain
from param import user_param

from FastSpeech2 import synthesize
import playsound
import uvicorn

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse

import IPython.display as ipd
app = FastAPI()

@app.get("/tts/demo/audio")
async def demo():

    return ipd.Audio('fine-tune-dataset/hws0120/HW-man/1.wav')

@app.post("/tts/train")
async def tts_train(target_inf: TTS_Dto.TTSTrainDto):
    # param을 설정하고 hparam에 통째로 넘겨 hp 활용
    param = user_param.UserParam(target_inf.caregiver_id, target_inf.voice_target)
    hp = hparams.hparam(param)

    # async로 별도의 thread로 이벤트 train
    # response
    FS2_trainer = FS2_e2e_train.FS2Train(hp)
    FS2_trainer.E2E_FS2_train() # async 적용

    #HG_trainer = HGtrain.HiFiGANTrain(param)
    #HG_trainer.train()          # async 적용, 학습 종료 시 request

    return {"response": "Fine-tune train Complete!"} 


@app.post("/tts/inference")
async def tts_inference(target_text: TTS_Dto.TTSInferenceDto):

    param = user_param.UserParam(target_text.caregiver_id, target_text.voice_target)
    hp = hparams.hparam(param)
    
    synthesizer = synthesize.Synthesizer(hp)
    wav_path = synthesizer.synthesize(target_text.msg)

    # wav 파일을 backend로 전달 (response)
    return FileResponse(wav_path)

@app.post("/tts/custom_inference")
async def tts_custom_synthesize():
    synthesize.custom_synthesizer()

    # wav 파일을 backend로 전달 (response)
    return {"response": "Custom Inference Complete!"} 


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        port=23910,
        host="0.0.0.0",
        reload=True
    )
