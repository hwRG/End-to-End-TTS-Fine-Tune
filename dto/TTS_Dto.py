from pydantic import BaseModel

class TTSTrainDto(BaseModel):
    caregiver_id: str   # hws0120
    voice_target: str   # HW-man

class TTSInferenceDto(BaseModel):
    msg: str            # 안녕하세요 반갑습니다
    caregiver_id: str   # hws0120
    voice_target: str   # HW-man
    