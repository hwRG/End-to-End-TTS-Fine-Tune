from FastSpeech2 import hparams, FS2_e2e_train
from HiFiGAN import train as HGtrain
from param import user_param

from FastSpeech2 import synthesize


def tts_train(caregiver_id, voice_target):
    # param을 설정하고 hparam에 통째로 넘겨 hp 활용
    param = user_param.UserParam(caregiver_id, voice_target)
    hp = hparams.hparam(param)

    FS2_trainer = FS2_e2e_train.FS2Train(hp)
    FS2_trainer.E2E_FS2_train() # async 적용
    #print(fs_trainer_futer)

    HG_trainer = HGtrain.HiFiGANTrain(param)
    HG_trainer.train()          # async 적용, 학습 종료 시 request

    return {"response": "Fine-tune train Complete!"} 


def tts_inference(caregiver_id, voice_target, msg):
    param = user_param.UserParam(caregiver_id, voice_target)
    hp = hparams.hparam(param)
    
    synthesizer = synthesize.Synthesizer(hp)
    wav_path = synthesizer.synthesize(msg)

    # wav 파일 경로를 backend로 전달 (response)
    return wav_path

    
if __name__ == "__main__":
    tts_train('pnylove12', 'NY-woman')
    #tts_inference('hws0120', 'HW-man', '안녕하세요 반갑습니다')