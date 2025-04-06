### python3 download_models.py

# Image swapper Torch model
### wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -P ~/.cache/torch/hub/checkpoints/
## OR 
### curl -o ~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

## THEN mv ~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth /Users/patrick/youtube_video/models/

## python3 talking_image_animation.py --source_image driving.jpg --audio my_voice.mp3 --output output.mp4

## python3 core/talking_image_animation_old.py --source_image driving.jpg --audio my_voice.wav --output output.mp4

## python3 talking_image_animation.py \
  --source_image path/to/source_image.jpg \
  --audio path/to/audio.mp3 \
  --output path/to/output.mp4 \
  --background path/to/background.jpg \
  --no_head_movement

# without head movements python3 talking_image_animation.py \
  --source_image portrait.jpg \
  --audio speech.mp3 \
  --output animated_video.mp4 \
  --no_head_movement

# with background image python3 talking_image_animation.py \
  --source_image portrait.jpg \
  --audio speech.mp3 \
  --output animated_video.mp4 \
  --background background.jpg
# with basic animations python3 talking_image_animation.py \
  --source_image portrait.jpg \
  --audio speech.mp3 \
  --output animated_video.mp4

## python3 core/video_generator.py --source source.jpg --driving driving.mp4 --output /Users/patrick/Desktop/result.mp4 --expression_intensity 0.8 --max_frames 20

## python3 core/video_generator.py --source source.jpg --driving driving.jpg --output /Users/patrick/Desktop/result.jpg 

## bttterresults python3 main.py --source source.jpg --driving driving.jpg --output result.jpg --max_frames 1 --fps 1 --expression_intensity 1.0 --no_stabilization

## python3 main.py --source source.jpg --driving driving.jpg --output result.mp4 --max_frames 1 --fps 1



Here's how to use the voice cloning system:

1. **Setup** - Install dependencies and create directories:
   ```bash
   pip install torch torchaudio librosa soundfile transformers pyaudio matplotlib scipy sklearn
   mkdir -p data/train data/val checkpoints voice_profiles
   ```

2. **Record voice sample**:
   ```bash
   python -c "from voice_cloner.voice_cloning_system import VoiceCloner; VoiceCloner().record_voice_sample(duration=15, output_path='my_voice.wav')"
   ```
   This records 15 seconds of your voice for cloning.

3. **Prepare training data**:
   - Create speaker directories with audiofiles:
     ```
     data/
     ├── your_name/
     │   ├── sample1.wav
     │   ├── sample2.wav
     ```
   - Run the preparation script:
     ```bash
     python -c "from voice_cloner.voice_cloning_system import VoiceCloner; VoiceCloner().prepare_training_data()"
     ```

4. **Train the model**:
   ```bash
   python -c "from voice_cloner.voice_cloning_system import VoiceCloner; VoiceCloner().train(epochs=50)"
   ```

5. **Create voice profile**:
   ```bash
   python -c "from voice_cloner.voice_cloning_system import VoiceCloner; VoiceCloner().create_voice_profile('my_voice.wav', 'narrator')"
   ```

6. **Generate speech for house tour**:
   ```bash
   # Create tour script (rooms separated by "## Room Name")
   echo -e "Welcome to this beautiful property.\n\n## Living Room\nNotice the spacious layout and natural light.\n\n## Kitchen\nThis gourmet kitchen features top appliances." > tour_script.txt
   
   # Generate audio segments
   python -c "from voice_cloner.voice_cloning_system import VoiceCloner; VoiceCloner().process_audio_for_tour(open('tour_script.txt').read(), 'narrator', segment=True)"
   ```

To simplify testing, you can use the command-line interface:
```bash
# Save the script as voice_cloning_system.py
python voice_cloning_system.py record --duration 15 --output my_voice.wav
python voice_cloning_system.py train --epochs 50
python voice_cloning_system.py profile create --audio my_voice.wav --name narrator
python voice_cloning_system.py tour --script tour_script.txt --voice narrator --segment
```

## for voice_cloner_grok.py 
# Prepare Dataset:
 python3 voice_cloner/voice_cloner_grok.py prepare \
    --audio_dir ./audio_files \
    --text_path ./my_voice.txt \
    --data_dir ./data \
    --config ./config.yaml \
    --test_split 0.1 \
    --val_split 0.1

# Train Model:
python3 voice_cloner/voice_cloner_grok.py train \
    --config config.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001

# Run Inference:
python3 voice_cloner/voice_cloner_grok.py infer \
    --checkpoint /path/to/model.pth \
    --text "Text to synthesize" \
    --reference_audio /path/to/reference.wav \
    --output_audio output.wav \
    --visualize

# Evaluate Model
python3 voice_cloner/voice_cloner_grok.py evaluate \
    --checkpoint /path/to/model.pth \
    --num_samples 10
    
# Real-time Conversion
python3 voice_cloner/voice_cloner_grok.py realtime \
    --checkpoint /path/to/model.pth \
    --reference_audio /path/to/reference.wav \
    --duration 60


## talking_head_animation
# Create a directory for models
mkdir -p models

# Download FOMM model

### working
# Create a Python script to download everything
cat > download_models.py << 'EOL'
from updated_model_downloader import PretrainedModelManager

# Download FOMM model
PretrainedModelManager.download_fomm_model('models/vox-cpk.pth')

# Download config file
PretrainedModelManager.download_config('config/vox-256.yaml')

# Download Wav2Lip model (optional, only if you want to use audio)
PretrainedModelManager.download_wav2lip_model('models/wav2lip_gan.pth')
EOL

# Run the download script
python download_fofm_model.py

###
python -c "from talking_head_animation import PretrainedModelManager; PretrainedModelManager.download_fomm_model('models/vox-cpk.pth')"

# Create config directory
mkdir -p config
You'll also need the YAML configuration file for the model. Create a file at config/vox-256.yaml with the appropriate configuration (you can find this in the FOMM GitHub repository).


## python3 core/animate_face.py --video_path input_video.mp4 --audio_path speech_audio.wav --output_path outputt_video.mp4 --wav2lip_model models/wav2lip.pth

### 
python3 core/new_face_animation_system_rule_based.py \
--video_path input_video.mp4 \
--audio_path speech_audio.wav \
--output_path animated_output.mp4

python3 core/new_face_animation_system_rule_based.py \
--video_path input_video.mp4 \
--audio_path speech_audio.wav \
--output_path animated_output.mp4