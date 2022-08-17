from pydantic import BaseModel

class TTSTrainDto(BaseModel):
    caregiver_id: str   # hws0120
    voice_target: str   # HW-man

class TTSInferenceDto(BaseModel):
    msg: str
    elderly_id: str
    