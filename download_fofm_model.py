from core.talking_head_animation import PretrainedModelManager

# Download FOMM model
PretrainedModelManager.download_fomm_model('models/vox-cpk.pth')

# Download config file
PretrainedModelManager.download_config('config/vox-256.yaml')

# Download Wav2Lip model (optional, only if you want to use audio)
PretrainedModelManager.download_wav2lip_model('models/wav2lip_gan.pth')
