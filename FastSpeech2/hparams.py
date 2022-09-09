import os

class hparam:
    def __init__(self, param=None):
        # async로부터 받은 param
        self.param = param

        self.restore_step = '300000_new'
        self.synthesize_step = '305000'

        # Default Sampling rate - 22050
        self.sampling_rate = 22050

        if self.param != None:
            self.base_dir = self.param.base_dir
            
            self.data_dir = self.param.data_dir # fine-tune-datset/ID
            self.target_dir = self.param.target_dir # ID/Speaker
            self.direct_dir = self.param.direct_dir # fine-tune-datset/ID/Speaker

            self.dataset = self.param.dataset # Speaker
            self.user_id = self.param.user_id # ID

            # Checkpoints and synthesis path
            self.preprocessed_path = os.path.join("./preprocessed/", self.target_dir)
            self.checkpoint_path = os.path.join("./ckpt/")
            self.eval_path = os.path.join("./eval/", self.target_dir)
            self.log_path = os.path.join("./log/", self.target_dir)
            self.test_path = os.path.join("./results/", self.target_dir)

            # Vocoder
            self.vocoder = 'hifigan'
            #self.vocoder_pretrained_model_name = self.dataset + "_g_00350000.pt" 
            self.vocoder_pretrained_model_name = "g_00350000.pt" # 600000 + 10000
            self.vocoder_pretrained_model_path = os.path.join(self.checkpoint_path, self.vocoder_pretrained_model_name)

            self.meta_name = "fine_tune_transcript.txt"
            self.textgrid_name = self.dataset + "textgrids.zip"


        ### set GPU number ###
        self.train_visible_devices = "0,1"
        self.synth_visible_devices = "1"

        # Text
        self.text_cleaners = ['korean_cleaners']


        # Audio and mel
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024

        self.max_wav_value = 32768.0
        self.n_mel_channels = 80
        self.mel_fmin = 0
        self.mel_fmax = 8000

        self.f0_min = 71.0
        self.f0_max = 792.8
        self.energy_min = 0.0
        self.energy_max = 283.72


        # FastSpeech 2
        self.encoder_layer = 4
        self.encoder_head = 2
        self.encoder_hidden = 256
        self.decoder_layer = 4
        self.decoder_head = 2
        self.decoder_hidden = 256
        self.fft_conv1d_filter_size = 1024
        self.fft_conv1d_kernel_size = (9, 1)
        self.encoder_dropout = 0.2
        self.decoder_dropout = 0.2

        self.variance_predictor_filter_size = 256
        self.variance_predictor_kernel_size = 3
        self.variance_predictor_dropout = 0.5

        self.max_seq_len = 1000



        # Optimizer
        self.batch_size = 8
        self.epochs = 630 # 5000 step
        self.n_warm_up_step = 4000
        self.grad_clip_thresh = 1.0
        self.acc_steps = 1

        self.betas = (0.9, 0.98)
        self.eps = 1e-9
        self.weight_decay = 0.


        # HiFi-GAN parameter
        self.resblock_kernel_sizes = [3,7,11]
        self.upsample_rates = [8,8,2,2]
        self.resblock = "1"
        self.upsample_rates = [8,8,2,2]
        self.upsample_kernel_sizes = [16,16,4,4]
        self.upsample_initial_channel = 512
        self.resblock_kernel_sizes = [3,7,11]
        self.resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]

        # Log-scaled duration
        self.log_offset = 1.

        # Save, log and synthesis
        self.save_step = 5000
        self.eval_step = 10000
        self.eval_size = 256
        self.log_step = 1000
        self.clear_Time = 20


