import librosa
import numpy as np
from typing import List, Tuple

class AudioVisualSynchronizer:
    def __init__(self):
        self.sample_rate = 44100
        self.hop_length = 512
        
    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract audio features for synchronization"""
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=80,
            hop_length=self.hop_length
        )
        
        return mel_spec
    
    def align_frames_to_audio(self, 
                            frames: List[np.ndarray], 
                            audio_features: np.ndarray) -> List[np.ndarray]:
        """Align video frames with audio features"""
        # Calculate frames per audio segment
        audio_segments = audio_features.shape[1]
        n_frames = len(frames)
        
        # Calculate alignment indices
        indices = np.linspace(0, audio_segments-1, n_frames).astype(int)
        
        return [frames[i] for i in indices]