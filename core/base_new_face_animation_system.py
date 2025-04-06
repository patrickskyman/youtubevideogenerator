## we can improve the script by 
'''
The key issue is that you're seeing a face detection overlay but not proper lip synchronization. Here are your options:
Option 1: Use a Pre-trained Viseme Model
This is the best approach. The code is looking for a file called "viseme_model.pth" but not finding it. You have a few options:

Find a pre-trained viseme model: Look for publicly available models that convert audio to facial landmarks. Models trained for lip-reading or speech animation could work.
Place it in the correct location: The model file should be in the same directory as your script.

Option 2: Train Your Own Model (More Complex)
To train your own model as suggested in the code comments:
pythonCopyanimation_system.train_viseme_model("path/to/audio_visual_dataset")
However, this would require:

A dataset with paired audio and video of people speaking
The implementation of the "viseme_model_training.py" script (which isn't included in your shared code)
GPU resources for training

Option 3: Improve the Rule-based System
The current implementation already has a rule-based system, but it needs improvements:

Improve viseme detection: The function generate_viseme_sequence in AudioProcessor class could be enhanced
Adjust viseme configurations: The mappings in viseme_configs in the VisemeGenerator class could be refined

Recommended Approach

First, debug the current rule-based system:

Add print statements in the _generate_visemes_from_audio_features method to see what visemes are being detected
Check if the intensity values are too low


Modify your command to add debug output:
bashCopypython3 core/new_face_animation_system.py \
--video_path input_video.mp4 \
--audio_path speech_audio.wav \
--output_path animated_output.mp4 --debug
(You'll need to add a debug parameter to the script)
Look for pre-trained models: Search for "facial animation model", "viseme model", or "audio-to-face model" that produces PyTorch models.
'''

import cv2
import numpy as np
import mediapipe as mp
import librosa
import torch
from typing import List, Dict, Optional, Tuple
import os
from tqdm import tqdm

class AudioProcessor:
    """Process audio to extract features for lip synchronization"""
    def __init__(self, sample_rate=16000, hop_length=512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Load pretrained model for phoneme detection if available
        self.phoneme_model = None
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.phoneme_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            print("Loaded phoneme recognition model")
        except:
            print("Warning: Transformers library not available. Using basic audio processing.")
    
    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extract additional features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Add delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Normalize features
        features = np.vstack([mfccs, mfcc_delta, mfcc_delta2, spectral_centroid, spectral_contrast])
        normalized = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        return normalized
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        audio, sr = librosa.load(audio_path, sr=None)
        return librosa.get_duration(y=audio, sr=sr)
    
    def generate_viseme_sequence(self, audio_path: str) -> List[Dict[str, float]]:
        """Generate sequence of visemes from audio"""
        # If phoneme model is available, use it for better viseme prediction
        if self.phoneme_model is not None:
            return self._generate_visemes_from_phonemes(audio_path)
        else:
            return self._generate_visemes_from_audio_features(audio_path)
    
    def _generate_visemes_from_phonemes(self, audio_path: str) -> List[Dict[str, float]]:
        """Generate visemes based on phoneme recognition"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Process through phoneme recognition model
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.phoneme_model(inputs.input_values)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
        
        # Get phoneme sequence with timing
        phoneme_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
        frame_time = 0.02  # 20ms per frame in wav2vec2
        
        # Define basic viseme mapping (Preston Blair's system)
        # This is a simplified mapping; a more detailed one would give better results
        phoneme_to_viseme = {
            # Map phonemes to visemes
            # Silence
            "sil": "rest",
            # Consonants
            "p": "bilabial",      # p, b, m
            "b": "bilabial",
            "m": "bilabial",
            "f": "labiodental",   # f, v
            "v": "labiodental",
            "th": "dental",       # th
            "t": "alveolar",      # t, d, n, l
            "d": "alveolar",
            "n": "alveolar",
            "l": "alveolar",
            "sh": "palato-alveolar", # sh, ch, j, zh
            "ch": "palato-alveolar",
            "j": "palato-alveolar",
            "zh": "palato-alveolar",
            "k": "velar",         # k, g, ng
            "g": "velar",
            "ng": "velar",
            # Vowels
            "aa": "open",         # ah, aa
            "ah": "open",
            "ae": "mid",          # eh, ae, ey
            "eh": "mid",
            "ey": "mid",
            "iy": "close",        # ee, i
            "ih": "close",
            "ow": "round",        # o, oh, aw
            "aw": "round",
            "oh": "round",
            "uw": "round-small",  # oo, w
            "w": "round-small",
        }
        
        # Map ID to phoneme
        id_to_phoneme = self.processor.tokenizer.decoder
        
        viseme_sequence = []
        current_viseme = "rest"
        current_start = 0
        
        # Process each frame
        for i, phoneme_id in enumerate(predicted_ids[0].numpy()):
            time = i * frame_time
            phoneme = id_to_phoneme.get(phoneme_id, "")
            
            # Get viseme for this phoneme
            viseme = phoneme_to_viseme.get(phoneme, "rest")
            
            # If viseme changed, add previous viseme to sequence
            if viseme != current_viseme and i > 0:
                intensity = np.max(phoneme_probs[i-1])  # Use probability as intensity
                
                viseme_sequence.append({
                    "viseme": current_viseme,
                    "start_time": current_start,
                    "duration": time - current_start,
                    "intensity": float(intensity)
                })
                
                current_viseme = viseme
                current_start = time
        
        # Add the last viseme
        if current_viseme != "rest":
            viseme_sequence.append({
                "viseme": current_viseme,
                "start_time": current_start,
                "duration": len(predicted_ids[0]) * frame_time - current_start,
                "intensity": 1.0
            })
        
        return viseme_sequence
    
    def _generate_visemes_from_audio_features(self, audio_path: str) -> List[Dict[str, float]]:
        """Generate visemes based on audio features when phoneme model is not available"""
        # Print debugging info
        print("Using rule-based viseme generation for audio:", audio_path)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Extract more detailed audio features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Get amplitude envelope to detect speech intensity
        hop_length = 512
        amplitude_envelope = np.array([max(audio[i:i+hop_length]) for i in range(0, len(audio), hop_length)])
        amplitude_times = np.array([i/sr*hop_length for i in range(len(amplitude_envelope))])
        
        # Threshold for detecting speech
        speech_threshold = 0.03
        
        # Generate viseme sequence
        viseme_sequence = []
        prev_viseme = "rest"
        
        for i, amp in enumerate(amplitude_envelope):
            current_time = amplitude_times[i]
            
            # Skip if amplitude is too low (silence)
            if amp < speech_threshold:
                if prev_viseme != "rest":
                    viseme_sequence.append({
                        "viseme": "rest",
                        "start_time": current_time,
                        "duration": 0.15,
                        "intensity": 0.1
                    })
                    prev_viseme = "rest"
                continue
                
            # Get MFCC segment for this window
            if i < mfccs.shape[1]:
                mfcc_segment = mfccs[:, i]
                
                # Select viseme based on MFCC features
                # Higher first coefficients often indicate open mouth sounds
                if mfcc_segment[1] > 100:
                    viseme = "wide_open"
                    intensity = 0.9
                elif mfcc_segment[2] > 50:
                    viseme = "open"
                    intensity = 0.7
                elif mfcc_segment[3] > 30:
                    viseme = "rounded"
                    intensity = 0.6
                else:
                    viseme = "slight_open"
                    intensity = 0.4
                    
                # Only add if different from previous or if time gap is significant
                if viseme != prev_viseme or len(viseme_sequence) == 0:
                    viseme_sequence.append({
                        "viseme": viseme,
                        "start_time": current_time,
                        "duration": 0.15,  # Fixed duration works better than trying to predict
                        "intensity": min(amp * 5, 1.0)  # Scale amplitude to reasonable intensity
                    })
                    prev_viseme = viseme
                    
                    # Debug output
                    print(f"Time: {current_time:.2f}, Viseme: {viseme}, Intensity: {min(amp * 5, 1.0):.2f}")
        
        return viseme_sequence


class VisemeGenerator:
    """Generate facial landmark configurations for different visemes (mouth shapes)"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Correct indices for mouth landmarks in MediaPipe Face Mesh
        self.outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91]
        self.mouth_indices = list(set(self.outer_lip_indices + self.inner_lip_indices))
        
        # Define viseme configurations (displacements from neutral position)
        # These are offset values to apply to the neutral landmarks
        # In a real system, these would be learned from data
        self.viseme_configs = {
            "rest": np.zeros((len(self.mouth_indices), 2)),
            "open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.05], [0, -0.07], [0, -0.08], [0, -0.08], [0, -0.07], [0, -0.05], [0, 0],
                [0, 0.05], [0, 0.07], [0, 0.08], [0, 0.07], [0, 0.05], [0, 0]
            ])[:len(self.mouth_indices)],
            "wide_open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.1], [0, -0.12], [0, -0.15], [0, -0.15], [0, -0.12], [0, -0.1], [0, 0],
                [0, 0.1], [0, 0.12], [0, 0.15], [0, 0.12], [0, 0.1], [0, 0]
            ])[:len(self.mouth_indices)],
            "rounded": np.array([
                [0.03, 0], [0.04, 0], [0.04, 0], [0.03, 0], [-0.03, 0], [-0.04, 0], [-0.04, 0], [-0.03, 0], [0, 0], [0, 0],
                [0.02, -0.02], [0.03, -0.03], [0, -0.04], [-0.03, -0.03], [-0.02, -0.02], [0, 0], [0, 0],
                [0.02, 0.02], [0.03, 0.03], [0, 0.04], [-0.03, 0.03], [-0.02, 0.02], [0, 0]
            ])[:len(self.mouth_indices)],
            "closed": np.zeros((len(self.mouth_indices), 2)),
            "slight_open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.03], [0, -0.04], [0, -0.05], [0, -0.05], [0, -0.04], [0, -0.03], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.04], [0, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            # Add more viseme configurations with proper offsets
            "bilabial": np.array([
                [0.02, 0], [0.03, 0], [0.03, 0], [0.02, 0], [-0.02, 0], [-0.03, 0], [-0.03, 0], [-0.02, 0], [0, 0], [0, 0],
                [0.01, -0.01], [0.01, -0.01], [0, -0.01], [-0.01, -0.01], [-0.01, -0.01], [0, 0], [0, 0],
                [0.01, 0.01], [0.01, 0.01], [0, 0.01], [-0.01, 0.01], [-0.01, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "labiodental": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.02], [0, -0.04], [0, -0.06], [0, -0.06], [0, -0.04], [0, -0.02], [0, 0],
                [0, 0.01], [0, 0.01], [0, 0.01], [0, 0.01], [0, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "dental": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.02], [0, -0.03], [0, -0.04], [0, -0.04], [0, -0.03], [0, -0.02], [0, 0],
                [0, 0.02], [0, 0.03], [0, 0.04], [0, 0.03], [0, 0.02], [0, 0]
            ])[:len(self.mouth_indices)],
            "alveolar": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.03], [0, -0.04], [0, -0.05], [0, -0.05], [0, -0.04], [0, -0.03], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.04], [0, 0.04], [0, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "palato-alveolar": np.array([
                [0.01, 0], [0.01, 0], [0.01, 0], [0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [0, 0], [0, 0],
                [0, -0.04], [0, -0.05], [0, -0.06], [0, -0.06], [0, -0.05], [0, -0.04], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.04], [0, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "velar": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.02], [0, -0.03], [0, -0.04], [0, -0.04], [0, -0.03], [0, -0.02], [0, 0],
                [0, 0.01], [0, 0.02], [0, 0.02], [0, 0.02], [0, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "mid": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.04], [0, -0.05], [0, -0.06], [0, -0.06], [0, -0.05], [0, -0.04], [0, 0],
                [0, 0.04], [0, 0.05], [0, 0.06], [0, 0.05], [0, 0.04], [0, 0]
            ])[:len(self.mouth_indices)],
            "close": np.array([
                [0.02, 0], [0.03, 0], [0.03, 0], [0.02, 0], [-0.02, 0], [-0.03, 0], [-0.03, 0], [-0.02, 0], [0, 0], [0, 0],
                [0, -0.01], [0, -0.01], [0, -0.02], [0, -0.02], [0, -0.01], [0, -0.01], [0, 0],
                [0, 0.01], [0, 0.01], [0, 0.02], [0, 0.01], [0, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "round": np.array([
                [0.03, 0], [0.04, 0], [0.04, 0], [0.03, 0], [-0.03, 0], [-0.04, 0], [-0.04, 0], [-0.03, 0], [0, 0], [0, 0],
                [0.02, -0.03], [0.03, -0.04], [0, -0.05], [-0.03, -0.04], [-0.02, -0.03], [0, 0], [0, 0],
                [0.02, 0.03], [0.03, 0.04], [0, 0.05], [-0.03, 0.04], [-0.02, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "round-small": np.array([
                [0.04, 0], [0.05, 0], [0.05, 0], [0.04, 0], [-0.04, 0], [-0.05, 0], [-0.05, 0], [-0.04, 0], [0, 0], [0, 0],
                [0.03, -0.02], [0.04, -0.03], [0, -0.03], [-0.04, -0.03], [-0.03, -0.02], [0, 0], [0, 0],
                [0.03, 0.02], [0.04, 0.03], [0, 0.03], [-0.04, 0.03], [-0.03, 0.02], [0, 0]
            ])[:len(self.mouth_indices)]
        }
    
    def get_neutral_landmarks(self, face_landmarks) -> np.ndarray:
        """Extract neutral mouth landmarks from face landmarks"""
        mouth_landmarks = []
        for idx in self.mouth_indices:
            landmark = face_landmarks.landmark[idx]
            mouth_landmarks.append([landmark.x, landmark.y])
        return np.array(mouth_landmarks)
    
    def apply_viseme(self, neutral_landmarks: np.ndarray, viseme: str, intensity: float = 1.0) -> np.ndarray:
        """Apply viseme deformation to neutral landmarks with intensity scaling"""
        if viseme not in self.viseme_configs:
            return neutral_landmarks
        
        # Get viseme configuration
        viseme_config = self.viseme_configs[viseme]
        
        # Scale by intensity
        viseme_offset = viseme_config * intensity
        
        # Apply offset to neutral landmarks
        return neutral_landmarks + viseme_offset


class FacialExpressionGenerator:
    """Generate facial expressions like emotions and blend them with speech"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Define landmark indices for different facial regions
        self.eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 
                               105, 66, 107, 55, 70, 46, 53, 52, 65, 55]
        self.eye_indices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
        self.nose_indices = [1, 168, 197, 195, 5, 4, 19, 94, 2]
        self.cheek_indices = [425, 205, 206, 429, 215, 50, 142, 348, 347, 280]
        
        # Define expression configurations as offsets from neutral
        self.expression_configs = {
            "neutral": np.zeros((468, 3)),  # No offset
            "happy": self._generate_happy_expression(),
            "sad": self._generate_sad_expression(),
            "angry": self._generate_angry_expression(),
            "surprised": self._generate_surprised_expression(),
            "thinking": self._generate_thinking_expression()
        }
    
    def _generate_happy_expression(self) -> np.ndarray:
        """Generate offset for happy expression"""
        expression = np.zeros((468, 3))
        
        # Raise cheeks
        for i, idx in enumerate(self.cheek_indices):
            expression[idx] = [0, -0.02, 0]  # Move cheeks up
        
        # Slight eyebrow raise
        for i, idx in enumerate(self.eyebrow_indices):
            if i < len(self.eyebrow_indices) // 2:  # Left eyebrow
                expression[idx] = [0, -0.01, 0]  # Raise slightly
            else:  # Right eyebrow
                expression[idx] = [0, -0.01, 0]  # Raise slightly
        
        return expression
    
    def _generate_sad_expression(self) -> np.ndarray:
        """Generate offset for sad expression"""
        expression = np.zeros((468, 3))
        
        # Lower inner eyebrows
        for i, idx in enumerate(self.eyebrow_indices):
            if i in [2, 3, 4, 12, 13, 14]:  # Inner parts of eyebrows
                expression[idx] = [0, 0.02, 0]  # Lower
        
        # Slight mouth corner down
        for idx in [61, 291]:  # Mouth corners
            expression[idx] = [0, 0.02, 0]  # Lower corners
        
        return expression
    
    def _generate_angry_expression(self) -> np.ndarray:
        """Generate offset for angry expression"""
        expression = np.zeros((468, 3))
        
        # Lower and bring eyebrows closer
        for i, idx in enumerate(self.eyebrow_indices):
            if i < len(self.eyebrow_indices) // 2:  # Left eyebrow
                if i < 3:  # Inner part
                    expression[idx] = [0.01, 0.02, 0]  # Down and in
                else:
                    expression[idx] = [0, 0.01, 0]  # Slightly down
            else:  # Right eyebrow
                if i > len(self.eyebrow_indices) - 4:  # Inner part
                    expression[idx] = [-0.01, 0.02, 0]  # Down and in
                else:
                    expression[idx] = [0, 0.01, 0]  # Slightly down
        
        return expression
    
    def _generate_surprised_expression(self) -> np.ndarray:
        """Generate offset for surprised expression"""
        expression = np.zeros((468, 3))
        
        # Raise eyebrows
        for i, idx in enumerate(self.eyebrow_indices):
            expression[idx] = [0, -0.03, 0]  # Raise
        
        # Widen eyes
        for i, idx in enumerate(self.eye_indices):
            if i % 2 == 0:  # Top eyelid points
                expression[idx] = [0, -0.02, 0]  # Move up
            else:  # Bottom eyelid points
                expression[idx] = [0, 0.02, 0]  # Move down
        
        return expression
    
    def _generate_thinking_expression(self) -> np.ndarray:
        """Generate offset for thinking expression"""
        expression = np.zeros((468, 3))
        
        # Raise one eyebrow
        for i, idx in enumerate(self.eyebrow_indices):
            if i < len(self.eyebrow_indices) // 2:  # Left eyebrow
                expression[idx] = [0, -0.02, 0]  # Raise
        
        # Slight head tilt
        for i in range(468):
            expression[i] += [0, 0, 0.005]  # Slight tilt
        
        return expression
    
    def apply_expression(self, 
                         landmarks: np.ndarray, 
                         expression: str, 
                         intensity: float = 1.0) -> np.ndarray:
        """Apply facial expression to landmarks with intensity scaling"""
        if expression not in self.expression_configs:
            return landmarks
        
        # Get expression configuration
        expression_config = self.expression_configs[expression]
        
        # Scale by intensity
        expression_offset = expression_config * intensity
        
        # Apply offset to landmarks
        new_landmarks = landmarks.copy()
        for i in range(min(len(landmarks), len(expression_offset))):
            new_landmarks[i] += expression_offset[i]
        
        return new_landmarks


class FaceAnimator:
    """Animate a face based on audio input and expression settings"""
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.viseme_generator = VisemeGenerator()
        self.expression_generator = FacialExpressionGenerator()
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For background replacement/masking
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Pre-trained viseme model
        self.viseme_model = None
        try:
            self.viseme_model = torch.load("viseme_model.pth")
            print("Loaded pre-trained viseme model")
        except:
            print("No pre-trained viseme model found. Using rule-based viseme generation.")
    
    def detect_landmarks(self, image):
        """Detect facial landmarks in image"""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        return results.multi_face_landmarks[0]
    
    def draw_landmarks(self, image, landmarks):
        """Draw facial landmarks on image"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    
    def landmarks_to_numpy(self, landmarks):
        """Convert MediaPipe landmarks to numpy array"""
        landmarks_array = np.zeros((468, 3))
        for i, landmark in enumerate(landmarks.landmark):
            if i < 468:  # Add this check to prevent index out of bounds
                landmarks_array[i] = [landmark.x, landmark.y, landmark.z]
        return landmarks_array
    
    def numpy_to_landmarks(self, landmarks_array):
        """Convert numpy array back to MediaPipe landmarks format"""
        from mediapipe.framework.formats import landmark_pb2
        
        landmarks = landmark_pb2.NormalizedLandmarkList()
        for i in range(landmarks_array.shape[0]):
            landmark = landmark_pb2.NormalizedLandmark()
            landmark.x = float(landmarks_array[i, 0])
            landmark.y = float(landmarks_array[i, 1])
            landmark.z = float(landmarks_array[i, 2])
            landmarks.landmark.append(landmark)
        return landmarks
    
    def blend_visemes(self, viseme_sequence, frame_time, neutral_mouth_landmarks):
        """Blend between visemes based on timing for smooth transitions"""
        # Find which visemes are active at this time
        active_visemes = []
        for v in viseme_sequence:
            start = v["start_time"]
            end = start + v["duration"]
            
            # Check if this viseme is active at the current frame time
            if start <= frame_time <= end:
                # Calculate how far into the viseme we are (0-1)
                progress = (frame_time - start) / v["duration"]
                
                # Apply easing for smoother transitions
                # Ease in/out - slow at start and end, faster in middle
                if progress < 0.5:
                    weight = 2 * progress * progress  # Ease in
                else:
                    progress = progress * 2 - 1
                    weight = 1 - (1 - progress) * (1 - progress)  # Ease out
                    weight = weight * 0.5 + 0.5  # Remap to 0.5-1 range
                
                active_visemes.append({
                    "viseme": v["viseme"],
                    "weight": weight * v["intensity"]
                })
        
        # If no active visemes, return to neutral
        if not active_visemes:
            return neutral_mouth_landmarks
        
        # Apply weighted blend of active visemes
        total_weight = sum(v["weight"] for v in active_visemes)
        if total_weight == 0:
            return neutral_mouth_landmarks
            
        # Normalize weights
        for v in active_visemes:
            v["weight"] /= total_weight
        
        # Start with neutral landmarks
        final_landmarks = neutral_mouth_landmarks.copy()
        
        # Apply each viseme with its weight
        for v in active_visemes:
            viseme_landmarks = self.viseme_generator.apply_viseme(
                neutral_mouth_landmarks, 
                v["viseme"], 
                v["weight"]
            )
            final_landmarks += (viseme_landmarks - neutral_mouth_landmarks) * v["weight"]
        
        return final_landmarks
    
    def apply_emotions(self, frame_time, emotions, neutral_landmarks):
        """Apply emotional expressions with proper timing and blending"""
        if not emotions:
            return neutral_landmarks
        
        # Find active emotions at this time
        active_emotions = []
        for emotion in emotions:
            start = emotion["start_time"]
            end = start + emotion["duration"]
            
            if start <= frame_time <= end:
                # Calculate blend weight based on timing
                if frame_time - start < 0.5:  # Fade in
                    weight = (frame_time - start) * 2
                elif end - frame_time < 0.5:  # Fade out
                    weight = (end - frame_time) * 2
                else:  # Full intensity
                    weight = 1.0
                    
                weight *= emotion["intensity"]
                active_emotions.append({
                    "emotion": emotion["emotion"],
                    "weight": weight
                })
        
        # If no active emotions, return neutral
        if not active_emotions:
            return neutral_landmarks
        
        # Normalize weights
        total_weight = sum(e["weight"] for e in active_emotions)
        for e in active_emotions:
            e["weight"] /= total_weight
        
        # Apply weighted blend of emotions
        final_landmarks = neutral_landmarks.copy()
        for e in active_emotions:
            emotion_landmarks = self.expression_generator.apply_expression(
                neutral_landmarks,
                e["emotion"],
                e["weight"]
            )
            # Blend this emotion
            final_landmarks += (emotion_landmarks - neutral_landmarks) * e["weight"]
        
        return final_landmarks
    
    def integrate_visemes_with_expression(self, mouth_landmarks, full_landmarks, neutral_landmarks):
        """Integrate modified mouth landmarks with full facial expression"""
        # Copy the full landmarks
        integrated_landmarks = full_landmarks.copy()
        
        # Replace only the mouth region with the viseme landmarks
        for i, idx in enumerate(self.viseme_generator.mouth_indices):
            # Only update x and y, keep z as is
            integrated_landmarks[idx, 0:2] = mouth_landmarks[i]
        
        return integrated_landmarks
    
    def animate_from_audio(self, 
                          video_path: str, 
                          audio_path: str, 
                          output_path: str, 
                          emotions: List[Dict] = None, 
                          background_path: str = None):
        """
        Create an animated face video synced with audio
        
        Args:
            video_path: Path to input video with face
            audio_path: Path to audio file for lip sync
            output_path: Path for output video
            emotions: List of emotions with timing info
                [{"emotion": "happy", "start_time": 1.0, "duration": 2.0, "intensity": 0.8}, ...]
            background_path: Path to optional background image
        """
        # Generate viseme sequence from audio
        # Add this debug printing
        print(f"Input video path: {os.path.abspath(video_path)}")
        print(f"Output path: {os.path.abspath(output_path)}")
        
        # Generate viseme sequence from audio
        viseme_sequence = self.audio_processor.generate_viseme_sequence(audio_path)
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get audio duration
        audio_duration = self.audio_processor.get_audio_duration(audio_path)
        
        # Load background if provided
        background = None
        if background_path:
            background = cv2.imread(background_path)
            if background is not None:
                background = cv2.resize(background, (width, height))
        
        # Prepare video writer with absolute path
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file {output_path}")
            return

        # Detect neutral landmarks from first frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            return
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get face landmarks from first frame
        neutral_face_landmarks = self.detect_landmarks(frame)
        if neutral_face_landmarks is None:
            print("No face detected in first frame")
            return
        
        # Convert to numpy for easier manipulation
        neutral_landmarks_np = self.landmarks_to_numpy(neutral_face_landmarks)
        
        # Extract neutral mouth landmarks
        neutral_mouth_landmarks = self.viseme_generator.get_neutral_landmarks(neutral_face_landmarks)
        
        # Process each frame
        with tqdm(total=total_frames, desc="Animating") as pbar:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current time in seconds
                frame_time = frame_index / fps
                
                # Stop if we've processed the entire audio
                if frame_time > audio_duration:
                    break
                
                # Detect face in current frame
                face_landmarks = self.detect_landmarks(frame)
                
                if face_landmarks:
                    # Convert to numpy
                    landmarks_np = self.landmarks_to_numpy(face_landmarks)
                    
                    # Blend visemes based on current time
                    mouth_landmarks = self.blend_visemes(
                        viseme_sequence, 
                        frame_time, 
                        neutral_mouth_landmarks
                    )
                    
                    # Apply emotional expressions if provided
                    if emotions:
                        landmarks_np = self.apply_emotions(
                            frame_time, 
                            emotions, 
                            neutral_landmarks_np
                        )
                    
                    # Integrate mouth movements with emotional expressions
                    final_landmarks = self.integrate_visemes_with_expression(
                        mouth_landmarks, 
                        landmarks_np, 
                        neutral_landmarks_np
                    )
                    
                    # Convert back to MediaPipe format
                    final_face_landmarks = self.numpy_to_landmarks(final_landmarks)
                    
                    # Handle background replacement if requested
                    if background is not None:
                        # Create segmentation mask
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.selfie_segmentation.process(rgb_frame)
                        mask = results.segmentation_mask
                        
                        # Threshold mask
                        condition = np.stack((mask,) * 3, axis=-1) > 0.6
                        
                        # Blend foreground with background
                        frame = np.where(condition, frame, background)
                    
                    # Draw modified landmarks
                    self.draw_landmarks(frame, final_face_landmarks)
                    
                    # Add debugging info
                    current_viseme = "none"
                    for v in viseme_sequence:
                        if v["start_time"] <= frame_time <= v["start_time"] + v["duration"]:
                            current_viseme = v["viseme"]
                            break
                    
                    cv2.putText(frame, f"Time: {frame_time:.2f}s", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Viseme: {current_viseme}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame to output
                out.write(frame)
                
                # Update progress
                frame_index += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        
        if not os.path.exists(output_path):
            print(f"Error: Output video file was not created at {output_path}")
            return
            
        print(f"Video created successfully at {output_path}")
        
        # Add audio to the final video
        self.add_audio_to_video(output_path, audio_path)
    
    def add_audio_to_video(self, video_path, audio_path):
        """Add audio to video using ffmpeg"""
        import subprocess
        import os
        
        # Check if files exist
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            return
            
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} does not exist")
            return
        
        # Create temporary output path
        output_with_audio = video_path.replace('.mp4', '_with_audio.mp4')
        
        # Print absolute paths for debugging
        print(f"Adding audio from {os.path.abspath(audio_path)} to {os.path.abspath(video_path)}")
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', os.path.abspath(video_path),
            '-i', os.path.abspath(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_with_audio
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute command
        try:
            subprocess.run(cmd, check=True)
            # Replace original file
            os.replace(output_with_audio, video_path)
            print(f"Successfully added audio to {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio: {e}")
            print("Output will not have audio. Please install ffmpeg or add audio manually.")

class EnhancedVisemeModel:
    """A neural network-based viseme model trained on audio-visual data"""
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load pretrained viseme model"""
        try:
            self.model = torch.load(model_path)
            self.model.eval()  # Set to evaluation mode
            print(f"Loaded viseme model from {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load viseme model: {e}")
            return False
    
    def predict_viseme_sequence(self, audio_features):
        """Predict viseme landmark positions from audio features"""
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Convert to torch tensor
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            mouth_landmarks = self.model(audio_tensor).squeeze(0).numpy()
        
        return mouth_landmarks


class FacialAnimationSystem:
    """Main class to manage the face animation pipeline"""
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.viseme_generator = VisemeGenerator()
        self.face_animator = FaceAnimator()
        self.enhanced_viseme_model = None
        
        # Try to load enhanced viseme model if available
        try:
            self.enhanced_viseme_model = EnhancedVisemeModel("viseme_model.pth")
            print("Using enhanced neural viseme model")
        except:
            print("Using rule-based viseme generation")
    
    def extract_emotional_cues(self, transcript: str) -> List[Dict]:
        """Extract emotional cues from speech transcript"""
        # This is a placeholder for a more sophisticated sentiment analysis
        # In a real system, you would use NLP models to extract emotions from text
        
        emotions = []
        emotion_keywords = {
            "happy": ["happy", "joy", "glad", "excited", "excellent", "wonderful", "great"],
            "sad": ["sad", "sorry", "disappointed", "unfortunately", "regret"],
            "angry": ["angry", "annoyed", "frustrating", "upset", "terrible"],
            "surprised": ["wow", "amazing", "incredible", "surprised", "shocked", "unexpected"],
            "thinking": ["hmm", "well", "let me think", "consider", "perhaps", "maybe"]
        }
        
        time = 0.0
        for line in transcript.split('\n'):
            if ' - ' in line:  # Format: "00:00 - Text"
                try:
                    timestamp, text = line.split(' - ', 1)
                    time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(':'))))
                except:
                    continue
            else:
                text = line
            
            # Check for emotion keywords
            detected_emotion = None
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        detected_emotion = emotion
                        break
                if detected_emotion:
                    break
            
            if detected_emotion:
                emotions.append({
                    "emotion": detected_emotion,
                    "start_time": time,
                    "duration": 2.0,  # Typical emotion duration
                    "intensity": 0.7
                })
        
        return emotions
    
    def animate_video(self, 
                    video_path: str, 
                    audio_path: str, 
                    output_path: str,
                    transcript_path: str = None, 
                    background_path: str = None):
        """Create animated talking face video with audio sync and emotions"""
        # Extract emotions from transcript if available
        emotions = None
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            emotions = self.extract_emotional_cues(transcript)
        
        # Animate using face animator
        self.face_animator.animate_from_audio(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            emotions=emotions,
            background_path=background_path
        )
    
    def train_viseme_model(self, data_dir: str, model_save_path: str = "viseme_model.pth"):
        """Train viseme model using data in the specified directory"""
        from viseme_model_training import train_viseme_model
        
        train_viseme_model(data_dir, epochs=50, batch_size=16, learning_rate=0.001)
    
    def generate_viseme_sequence(self, audio_path: str) -> List[Dict]:
        """Generate viseme sequence from audio"""
        return self.audio_processor.generate_viseme_sequence(audio_path)
    
    def preview_visemes(self, video_path: str, viseme_sequence: List[Dict]):
        """Preview visemes overlaid on video frames"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get neutral face from first frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            return
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Detect neutral face
        neutral_face_landmarks = self.face_animator.detect_landmarks(frame)
        if neutral_face_landmarks is None:
            print("No face detected in first frame")
            return
        
        # Extract neutral mouth landmarks
        neutral_mouth_landmarks = self.viseme_generator.get_neutral_landmarks(neutral_face_landmarks)
        
        # Create preview window
        cv2.namedWindow("Viseme Preview", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current time
            current_time = frame_count / fps
            
            # Find active visemes at this time
            active_viseme = "rest"
            for v in viseme_sequence:
                if v["start_time"] <= current_time <= v["start_time"] + v["duration"]:
                    active_viseme = v["viseme"]
                    break
            
            # Display frame with viseme info
            cv2.putText(frame, f"Time: {current_time:.2f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Viseme: {active_viseme}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Viseme Preview", frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Animation System')
    parser.add_argument('--video_path', required=True, help='Path to input video file')
    parser.add_argument('--audio_path', required=True, help='Path to audio file for lip syncing')
    parser.add_argument('--output_path', required=True, help='Path for output video')
    parser.add_argument('--transcript_path', help='Path to transcript file for emotion extraction')
    parser.add_argument('--background_path', help='Path to background image')
    
    args = parser.parse_args()
    
    # Initialize face animation system
    animation_system = FacialAnimationSystem()
    
    # Animate video with provided arguments
    animation_system.animate_video(
        video_path=args.video_path,
        audio_path=args.audio_path,
        output_path=args.output_path,
        transcript_path=args.transcript_path,
        background_path=args.background_path
    )
    # Example 3: Training a custom viseme model
    # Uncomment to train (requires dataset)
    # animation_system.train_viseme_model("path/to/audio_visual_dataset")
    
    # Example 4: Preview visemes for an audio file
    viseme_sequence = animation_system.generate_viseme_sequence("speech_audio.wav")
    animation_system.preview_visemes("input_video.mp4", viseme_sequence)