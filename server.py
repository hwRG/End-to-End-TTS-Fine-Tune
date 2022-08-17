from enum import Enum

from fastapi import FastAPI

from dto.TTS_dto import TTSTrainDto, TTSInferenceDto

# uvicorn 2enum:app --reload

# str을 상속하여 값이 모두 str이어야 한다고 인지하고 제대로 렌더링
class ModelName(str, Enum):
    fastspeech2 = "fastspeech2"
    hifi_gan = "hifi-gan"
    tacotron2 = "tacotron2"
    waveglow = "waveglow"

app = FastAPI()


@app.get("/tts/train")
async def tts_train(target_inf: TTSTrainDto):
    if model_name == ModelName.fastspeech2:
        return {"model_name": model_name, "message": "FastSpeech2 Ckpt load Successed"}

    return {"model_name": model_name, "message": "Best Performance"}


@app.get("/tts/inference")
async def tts_inference(model_name: TTSInferenceDto):
    if model_name == ModelName.fastspeech2:
        return {"model_name": model_name, "message": "FastSpeech2 Ckpt load Successed"}

    if model_name.value == "hifi-gan":
        return {"model_name": model_name, "message": "HiFi-GAN upload OK"}

    return {"model_name": model_name, "message": "Best Performance"}

    

from model.ai_chatbot import getChatbotModel
from model.nearby_chatbot import NearbyLogic
from dto.chatting_dto import ChattingDto
from fastapi import FastAPI
import uvicorn

app = FastAPI()
AIChatModel = getChatbotModel([
                                "chatterbot.corpus.korean"
                            ])

RuleChatModel = NearbyLogic()

@app.post('/chat')
async def chatbot_response(message: ChattingDto):
    _u = message.msg
    elderly_id = message.elderly_id
    print(_u)
    res = RuleChatModel.process(_u, elderly_id)
    print(res)
    return { "response": str(res) }
    
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
        host="0.0.0.0",
        reload=True
    )