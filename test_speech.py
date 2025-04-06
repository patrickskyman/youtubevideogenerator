import os

# Test file existence
audio_path = "my_voice.wav"
print(f"File exists: {os.path.exists(audio_path)}")

# Try different loading methods
try:
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    print(f"Loaded with soundfile: {len(audio)} samples, {sr}Hz")
except Exception as e:
    print(f"SoundFile error: {e}")

try:
    import librosa
    audio, sr = librosa.load(audio_path, sr=None)
    print(f"Loaded with librosa: {len(audio)} samples, {sr}Hz")
except Exception as e:
    print(f"Librosa error: {e}")

try:
    from scipy.io import wavfile
    sr, audio = wavfile.read(audio_path)
    print(f"Loaded with scipy: {len(audio)} samples, {sr}Hz")
except Exception as e:
    print(f"Scipy error: {e}")