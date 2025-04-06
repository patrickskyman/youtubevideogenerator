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
        print(f"Generating viseme sequence for {audio_path}")
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return []
            
        # Check audio characteristics
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = librosa.get_duration(y=audio, sr=sr)
            max_amplitude = np.max(np.abs(audio))
            
            print(f"Audio duration: {duration:.2f}s, Sample rate: {sr}Hz, Max amplitude: {max_amplitude:.4f}")
            
            if max_amplitude < 0.01:
                print("Warning: Audio volume is very low. This may affect viseme detection.")
            
            if duration < 0.5:
                print("Warning: Audio is very short. May not contain enough speech.")
        except Exception as e:
            print(f"Error analyzing audio: {e}")
        
        # If phoneme model is available, use it for better viseme prediction
        if self.phoneme_model is not None:
            try:
                print("Using phoneme-based viseme generation")
                visemes = self._generate_visemes_from_phonemes(audio_path)
                if visemes and len(visemes) > 3:
                    return visemes
                else:
                    print("Phoneme-based method produced too few visemes. Falling back to audio features.")
            except Exception as e:
                print(f"Error in phoneme-based viseme generation: {e}")
                print("Falling back to audio feature-based method.")
        
        # Use audio features as fallback
        try:
            print("Using audio feature-based viseme generation")
            return self._generate_visemes_from_audio_features(audio_path)
        except Exception as e:
            print(f"Error in audio feature-based viseme generation: {e}")
            print("Returning minimal fallback viseme sequence")
            
            # Create a minimal fallback sequence
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            return [
                {"viseme": "rest", "start_time": 0.0, "duration": 0.2, "intensity": 0.5},
                {"viseme": "open", "start_time": 0.2, "duration": 0.3, "intensity": 0.7},
                {"viseme": "slight_open", "start_time": 0.5, "duration": 0.3, "intensity": 0.6},
                {"viseme": "rest", "start_time": 0.8, "duration": 0.2, "intensity": 0.5},
                {"viseme": "round", "start_time": 1.0, "duration": 0.3, "intensity": 0.7},
                {"viseme": "rest", "start_time": 1.3, "duration": duration - 1.3, "intensity": 0.5}
            ]
    
    def _generate_visemes_from_phonemes(self, audio_path: str) -> List[Dict[str, float]]:
        """Generate visemes based on phoneme recognition"""

        last_intensity = 0.0
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Process through phoneme recognition model
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.phoneme_model(inputs.input_values)
            logits = outputs.logits
        
        # Get phoneme sequence with timing
        phoneme_probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
        predicted_ids = torch.argmax(logits, dim=-1)[0].numpy()
        
        # Map IDs to phoneme tokens
        id_to_phoneme = self.processor.tokenizer.convert_ids_to_tokens
        
        # Define basic viseme mapping (Preston Blair's system)
        phoneme_to_viseme = {
            # Silence and unknown tokens
            "<pad>": "rest",
            "<s>": "rest", 
            "</s>": "rest",
            "<unk>": "rest",
            # Consonants
            "P": "bilabial",
            "B": "bilabial",
            "M": "bilabial",
            "F": "labiodental",
            "V": "labiodental",
            "TH": "dental",
            "DH": "dental",
            "T": "alveolar",
            "D": "alveolar",
            "N": "alveolar",
            "L": "alveolar",
            "S": "dental",
            "Z": "dental",
            "SH": "palato-alveolar",
            "CH": "palato-alveolar",
            "JH": "palato-alveolar",
            "ZH": "palato-alveolar",  
            "K": "velar",
            "G": "velar",
            "NG": "velar",
            # Vowels
            "AA": "open",
            "AE": "wide_open",
            "AH": "open",
            "AO": "round",
            "AW": "round",
            "AY": "wide_open",
            "EH": "mid",
            "ER": "rounded",
            "EY": "mid",
            "IH": "slight_open",
            "IY": "close",
            "OW": "round",
            "OY": "round",
            "UH": "round-small",
            "UW": "round-small",
        }
        
        # Convert lowercase phonemes to visemes too
        lowercase_mapping = {k.lower(): v for k, v in phoneme_to_viseme.items()}
        phoneme_to_viseme.update(lowercase_mapping)
        
        # Add individual letters as fallbacks
        for letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if letter not in phoneme_to_viseme:
                # Map to reasonable defaults based on typical pronunciation
                if letter.lower() in 'aeiou':
                    phoneme_to_viseme[letter] = "open"
                elif letter.lower() in 'pbm':
                    phoneme_to_viseme[letter] = "bilabial"
                elif letter.lower() in 'fv':
                    phoneme_to_viseme[letter] = "labiodental"
                elif letter.lower() in 'tdnls':
                    phoneme_to_viseme[letter] = "alveolar"
                else:
                    phoneme_to_viseme[letter] = "slight_open"
        
        frame_time = 0.02  # 20ms per frame in wav2vec2
        viseme_sequence = []
        current_viseme = "rest"
        current_start = 0
        
        # Add debug print to see phoneme IDs
        print(f"First 20 phoneme IDs: {predicted_ids[:20]}")
        
        # Process each frame
        for i, phoneme_id in enumerate(predicted_ids):
            try:
                time = i * frame_time
                phoneme = id_to_phoneme(phoneme_id)
                
                # Debug print for first few tokens
                if i < 20:
                    print(f"Token ID {phoneme_id} -> '{phoneme}'")
                    
                # Get viseme for this phoneme
                viseme = phoneme_to_viseme.get(phoneme, "rest")
                intensity = float(np.max(phoneme_probs[i]))  # Use probability as intensity
                
                # If viseme changed or intensity changed significantly, add to sequence
                if (viseme != current_viseme or abs(intensity - last_intensity) > 0.2) and i > 0:
                    viseme_sequence.append({
                        "viseme": current_viseme,
                        "start_time": current_start,
                        "duration": time - current_start,
                        "intensity": last_intensity
                    })
                    
                    current_viseme = viseme
                    current_start = time
                
                last_intensity = intensity
            except Exception as e:
                print(f"Error processing phoneme at frame {i}: {e}")
                continue
        
        # Add the last viseme
        if current_viseme != "rest":
            viseme_sequence.append({
                "viseme": current_viseme,
                "start_time": current_start,
                "duration": len(predicted_ids) * frame_time - current_start,
                "intensity": last_intensity
            })
        
        # Print summary of detected visemes
        print(f"Generated {len(viseme_sequence)} visemes from phonemes")
        viseme_counts = {}
        for v in viseme_sequence:
            viseme_counts[v["viseme"]] = viseme_counts.get(v["viseme"], 0) + 1
        print(f"Viseme distribution: {viseme_counts}")

        # Add this at the beginning of the function
        last_intensity = 0.0
        
        return viseme_sequence
    
    def _generate_visemes_from_audio_features(self, audio_path: str) -> List[Dict[str, float]]:
        """Generate visemes based on audio features when phoneme model is not available"""
        print("Using enhanced rule-based viseme generation for audio:", audio_path)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Extract detailed audio features
        hop_length = 512
        n_fft = 2048
        
        # MFCC features - good for phoneme distinction
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Spectral features - help distinguish between different mouth shapes
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Energy features - help determine intensity of pronunciation
        rms_energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Zero crossing rate - help distinguish fricatives (s, f, th)
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)[0]
        
        # Compute timestamps for each frame
        timestamps = librosa.times_like(mfccs[0], sr=sr, hop_length=hop_length)
        
        # Thresholds for speech detection and viseme classification
        # Lower this threshold for more viseme detection
        energy_threshold = 0.01
        min_viseme_duration = 0.05  # Reduced minimum duration
        
        # Generate viseme sequence
        viseme_sequence = []
        prev_viseme = "rest"
        viseme_start_time = 0
        active_frames = []
        
        # Debug - check if we have energy in the audio
        print(f"Audio energy range: {rms_energy.min():.4f} to {rms_energy.max():.4f}")
        print(f"Number of frames above threshold: {np.sum(rms_energy > energy_threshold)}/{len(rms_energy)}")
        
        for i in range(len(timestamps)):
            # Skip frames with too low energy (silence)
            if rms_energy[i] < energy_threshold:
                if prev_viseme != "rest" and len(active_frames) > 0:
                    # End current viseme
                    duration = timestamps[i] - viseme_start_time
                    if duration >= min_viseme_duration:
                        # Calculate average intensity based on collected frames
                        avg_intensity = np.mean([rms_energy[j] for j in active_frames]) * 5  # Scale up
                        
                        viseme_sequence.append({
                            "viseme": prev_viseme,
                            "start_time": viseme_start_time,
                            "duration": duration,
                            "intensity": min(float(avg_intensity), 1.0)
                        })
                    
                    # Reset for "rest" viseme
                    prev_viseme = "rest"
                    viseme_start_time = timestamps[i]
                    active_frames = []
                continue
            
            # Extract features for this frame
            mfcc_frame = mfccs[:, i]
            centroid = spectral_centroid[i]
            bandwidth = spectral_bandwidth[i]
            zcr_val = zcr[i]
            energy = rms_energy[i]
            
            # Determine viseme based on audio characteristics
            # This mapping is the core of the improvement
            if energy > 0.1:  # High energy sounds
                if centroid > 4000 and zcr_val > 0.15:
                    # High frequency + high ZCR = fricatives like 's', 'f', 'th'
                    viseme = "dental" if zcr_val > 0.2 else "labiodental"
                elif centroid < 1500:
                    # Low frequency = open vowels like 'ah', 'aa'
                    viseme = "wide_open" if energy > 0.15 else "open"
                else:
                    # Mid frequency
                    viseme = "open"
            elif centroid > 3000:
                # Mid energy, high frequency = 't', 'd', etc.
                viseme = "alveolar"
            elif centroid < 1000:
                # Mid energy, low frequency = 'o', 'u'
                viseme = "round" if bandwidth < 1500 else "rounded"
            elif zcr_val > 0.1:
                # Higher zero crossing rate = consonants
                viseme = "bilabial" if centroid < 2000 else "palato-alveolar"
            else:
                # Default for other sounds
                viseme = "slight_open"
            
            # Handle transitions between visemes
            if viseme != prev_viseme:
                # Save previous viseme if it lasted long enough
                if prev_viseme != "rest" and len(active_frames) > 0:
                    duration = timestamps[i] - viseme_start_time
                    if duration >= min_viseme_duration:
                        # Calculate average intensity
                        avg_intensity = np.mean([rms_energy[j] for j in active_frames]) * 5
                        
                        viseme_sequence.append({
                            "viseme": prev_viseme,
                            "start_time": viseme_start_time,
                            "duration": duration,
                            "intensity": min(float(avg_intensity), 1.0)
                        })
                
                # Start new viseme
                prev_viseme = viseme
                viseme_start_time = timestamps[i]
                active_frames = [i]
            else:
                # Continue current viseme
                active_frames.append(i)
        
        # Add the last viseme if there is one
        if prev_viseme != "rest" and len(active_frames) > 0:
            duration = timestamps[-1] - viseme_start_time
            if duration >= min_viseme_duration:
                avg_intensity = np.mean([rms_energy[j] for j in active_frames]) * 5
                
                viseme_sequence.append({
                    "viseme": prev_viseme,
                    "start_time": viseme_start_time,
                    "duration": duration,
                    "intensity": min(float(avg_intensity), 1.0)
                })
        
        # Add debugging info
        print(f"Generated {len(viseme_sequence)} visemes from audio")
        for i, v in enumerate(viseme_sequence[:10]):  # Print first 10 for debugging
            print(f"{i+1}. {v['viseme']} from {v['start_time']:.2f}s to {v['start_time'] + v['duration']:.2f}s (intensity: {v['intensity']:.2f})")
        
        # Ensure there are enough visemes
        if len(viseme_sequence) < 5:
            print("Warning: Few visemes detected. Adding fallback visemes...")
            # If very few visemes were detected, create synthetic ones based on RMS energy patterns
            if len(rms_energy) > 0:
                # Find peaks in energy that might correspond to syllables
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(rms_energy, height=energy_threshold, distance=int(0.2 * sr / hop_length))
                
                for p in peaks[:20]:  # Use at most 20 peaks
                    time = timestamps[p]
                    energy_val = rms_energy[p]
                    
                    # Choose viseme based on energy level and position
                    if energy_val > 0.1:
                        viseme_type = "open"
                    elif energy_val > 0.05:
                        viseme_type = "slight_open"
                    else:
                        viseme_type = "bilabial"
                    
                    # Add synthetic viseme
                    viseme_sequence.append({
                        "viseme": viseme_type,
                        "start_time": time - 0.1,
                        "duration": 0.2,
                        "intensity": min(float(energy_val * 5), 1.0)
                    })
                
                print(f"Added {len(peaks)} synthetic visemes from energy peaks")
        
        # Sort by start time
        viseme_sequence.sort(key=lambda x: x["start_time"])
        
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
        

        # Generate default viseme configurations based on mouth indices
        self.generate_default_viseme_configs()

        # Define viseme configurations (displacements from neutral position)
        # These are offset values to apply to the neutral landmarks
        # In a real system, these would be learned from data
        # Define viseme configurations (displacements from neutral position)
        self.viseme_configs = {
            "rest": np.zeros((len(self.mouth_indices), 2)),
            "open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.05], [0, -0.07], [0, -0.08], [0, -0.08], [0, -0.07], [0, -0.05], [0, 0],
                [0, 0.05], [0, 0.07], [0, 0.08], [0, 0.07], [0, 0.05], [0, 0]
            ])[:len(self.mouth_indices)],
            "wide_open": np.array([
                [0.01, 0], [0.02, 0], [0.02, 0], [0.01, 0], [-0.01, 0], [-0.02, 0], [-0.02, 0], [-0.01, 0], [0, 0], [0, 0],
                [0, -0.12], [0, -0.15], [0, -0.18], [0, -0.18], [0, -0.15], [0, -0.12], [0, 0],
                [0, 0.10], [0, 0.12], [0, 0.15], [0, 0.12], [0, 0.10], [0, 0]
            ])[:len(self.mouth_indices)],
            "rounded": np.array([
                [0.04, 0], [0.05, 0], [0.05, 0], [0.04, 0], [-0.04, 0], [-0.05, 0], [-0.05, 0], [-0.04, 0], [0, 0], [0, 0],
                [0.03, -0.02], [0.04, -0.03], [0, -0.04], [-0.04, -0.03], [-0.03, -0.02], [0, 0], [0, 0],
                [0.03, 0.02], [0.04, 0.03], [0, 0.04], [-0.04, 0.03], [-0.03, 0.02], [0, 0]
            ])[:len(self.mouth_indices)],
            "closed": np.zeros((len(self.mouth_indices), 2)),
            "slight_open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.03], [0, -0.04], [0, -0.05], [0, -0.05], [0, -0.04], [0, -0.03], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.04], [0, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "bilabial": np.array([
                [0.03, 0], [0.04, 0], [0.04, 0], [0.03, 0], [-0.03, 0], [-0.04, 0], [-0.04, 0], [-0.03, 0], [0, 0], [0, 0],
                [0.02, -0.01], [0.02, -0.01], [0, -0.01], [-0.02, -0.01], [-0.02, -0.01], [0, 0], [0, 0],
                [0.02, 0.01], [0.02, 0.01], [0, 0.01], [-0.02, 0.01], [-0.02, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "labiodental": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.02], [0, -0.04], [0, -0.06], [0, -0.06], [0, -0.04], [0, -0.02], [0, 0],
                [0, 0.01], [0, 0.01], [0, 0.01], [0, 0.01], [0, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "dental": np.array([
                [-0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [0.01, 0], [0.01, 0], [0.01, 0], [0.01, 0], [0, 0], [0, 0],
                [0, -0.03], [0, -0.04], [0, -0.05], [0, -0.05], [0, -0.04], [0, -0.03], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.04], [0, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "alveolar": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.04], [0, -0.05], [0, -0.06], [0, -0.06], [0, -0.05], [0, -0.04], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.04], [0, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "palato-alveolar": np.array([
                [0.01, 0], [0.01, 0], [0.01, 0], [0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [0, 0], [0, 0],
                [0, -0.05], [0, -0.06], [0, -0.07], [0, -0.07], [0, -0.06], [0, -0.05], [0, 0],
                [0, 0.04], [0, 0.05], [0, 0.06], [0, 0.05], [0, 0.04], [0, 0]
            ])[:len(self.mouth_indices)],
            "velar": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.03], [0, -0.04], [0, -0.05], [0, -0.05], [0, -0.04], [0, -0.03], [0, 0],
                [0, 0.02], [0, 0.03], [0, 0.03], [0, 0.03], [0, 0.02], [0, 0]
            ])[:len(self.mouth_indices)],
            "mid": np.array([
                [0.01, 0], [0.01, 0], [0.01, 0], [0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [-0.01, 0], [0, 0], [0, 0],
                [0, -0.05], [0, -0.06], [0, -0.07], [0, -0.07], [0, -0.06], [0, -0.05], [0, 0],
                [0, 0.05], [0, 0.06], [0, 0.07], [0, 0.06], [0, 0.05], [0, 0]
            ])[:len(self.mouth_indices)],
            "close": np.array([
                [0.02, 0], [0.03, 0], [0.03, 0], [0.02, 0], [-0.02, 0], [-0.03, 0], [-0.03, 0], [-0.02, 0], [0, 0], [0, 0],
                [0, -0.01], [0, -0.02], [0, -0.02], [0, -0.02], [0, -0.02], [0, -0.01], [0, 0],
                [0, 0.01], [0, 0.02], [0, 0.02], [0, 0.02], [0, 0.01], [0, 0]
            ])[:len(self.mouth_indices)],
            "round": np.array([
                [0.04, 0], [0.05, 0], [0.05, 0], [0.04, 0], [-0.04, 0], [-0.05, 0], [-0.05, 0], [-0.04, 0], [0, 0], [0, 0],
                [0.03, -0.03], [0.04, -0.04], [0, -0.05], [-0.04, -0.04], [-0.03, -0.03], [0, 0], [0, 0],
                [0.03, 0.03], [0.04, 0.04], [0, 0.05], [-0.04, 0.04], [-0.03, 0.03], [0, 0]
            ])[:len(self.mouth_indices)],
            "round-small": np.array([
                [0.05, 0], [0.06, 0], [0.06, 0], [0.05, 0], [-0.05, 0], [-0.06, 0], [-0.06, 0], [-0.05, 0], [0, 0], [0, 0],
                [0.04, -0.02], [0.05, -0.03], [0, -0.03], [-0.05, -0.03], [-0.04, -0.02], [0, 0], [0, 0],
                [0.04, 0.02], [0.05, 0.03], [0, 0.03], [-0.05, 0.03], [-0.04, 0.02], [0, 0]
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
            print(f"Warning: Unknown viseme '{viseme}', falling back to rest")
            return neutral_landmarks
        
        # Get viseme configuration
        viseme_config = self.viseme_configs[viseme]
        
        # Check if shapes match and resize if needed
        if viseme_config.shape != neutral_landmarks.shape:
            # Only print this warning once per viseme
            if not hasattr(self, '_warned_visemes') or viseme not in self._warned_visemes:
                if not hasattr(self, '_warned_visemes'):
                    self._warned_visemes = set()
                self._warned_visemes.add(viseme)
                print(f"Warning: Viseme '{viseme}' shape mismatch - config: {viseme_config.shape}, landmarks: {neutral_landmarks.shape}")
                print(f"This message will be shown once per viseme type")
            
            # Create a correctly sized viseme config filled with zeros
            resized_config = np.zeros(neutral_landmarks.shape)
            
            # Copy available values, limited by the smaller shape
            min_rows = min(viseme_config.shape[0], resized_config.shape[0])
            min_cols = min(viseme_config.shape[1], resized_config.shape[1])
            resized_config[:min_rows, :min_cols] = viseme_config[:min_rows, :min_cols]
            
            viseme_config = resized_config
        
        # Increase intensity multiplier for more visible movement (adjust as needed)
        amplified_intensity = intensity * 2.5  # Amplify the movement
        
        # Scale by intensity
        viseme_offset = viseme_config * amplified_intensity
        
        # Apply offset to neutral landmarks
        return neutral_landmarks + viseme_offset

    def generate_default_viseme_configs(self):
        """Generate default viseme configurations based on the number of mouth landmarks"""
        num_landmarks = len(self.mouth_indices)
        
        # Enhanced viseme configurations with larger offsets for more visible movement
        self.viseme_configs = {
            "rest": np.zeros((num_landmarks, 2)),
            
            # Open mouth (like 'ah') - increased vertical movement
            "open": np.array([[0, 0.05] if i % 2 == 0 else [0, -0.05] for i in range(num_landmarks)]),
            
            # Wide open mouth (like 'aa') - even more vertical movement
            "wide_open": np.array([[0, 0.08] if i % 2 == 0 else [0, -0.08] for i in range(num_landmarks)]),
            
            # Slight open (like 'ih') - subtle but noticeable
            "slight_open": np.array([[0, 0.03] if i % 2 == 0 else [0, -0.03] for i in range(num_landmarks)]),
            
            # Round mouth (like 'oh') - more pronounced rounding
            "round": np.array([[-0.03 if i < num_landmarks//2 else 0.03, 0.02] if i % 2 == 0 
                            else [0.03 if i < num_landmarks//2 else -0.03, -0.02] for i in range(num_landmarks)]),
            
            # Bilabial (like 'p', 'b', 'm') - more closure
            "bilabial": np.array([[0, 0.02] if i % 2 == 0 else [0, -0.02] for i in range(num_landmarks)]),
            
            # Labiodental (like 'f', 'v') - more pronounced
            "labiodental": np.array([[0, 0] if i < num_landmarks//2 else [0, 0.03] for i in range(num_landmarks)]),
            
            # Dental (like 'th') - more movement
            "dental": np.array([[0, 0] if i < num_landmarks//3 else [0, 0.03] for i in range(num_landmarks)]),
            
            # Alveolar (like 't', 'd', 'n') - more movement
            "alveolar": np.array([[0, 0.03] if i % 3 == 0 else [0, 0] for i in range(num_landmarks)]),
        }
        
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
        # Check if we have valid inputs
        if not viseme_sequence or len(viseme_sequence) == 0:
            print(f"Warning: Empty viseme sequence at frame_time {frame_time}")
            return neutral_mouth_landmarks
        
        # Find which visemes are active at this time
        active_visemes = []
        for v in viseme_sequence:
            start = v["start_time"]
            end = start + v["duration"]
            
            # Add transition time
            transition_time = 0.05  # 50ms transition
            extended_start = max(0, start - transition_time)
            extended_end = end + transition_time
            
            # Check if this viseme is active at the current frame time
            if extended_start <= frame_time <= extended_end:
                # Calculate base weight based on position within the viseme
                if frame_time < start:
                    # In lead-in transition
                    weight = (frame_time - extended_start) / transition_time
                elif frame_time > end:
                    # In lead-out transition
                    weight = 1.0 - (frame_time - end) / transition_time
                else:
                    # In full viseme
                    progress = (frame_time - start) / v["duration"]
                    # Apply easing curve for smoother movement
                    if progress < 0.2:
                        weight = progress / 0.2  # Ease in
                    elif progress > 0.8:
                        weight = 1.0 - (progress - 0.8) / 0.2  # Ease out
                    else:
                        weight = 1.0  # Full strength
                
                # Adjust weight by viseme intensity and increase it
                final_weight = weight * v["intensity"] * 1.5  # Increase intensity
                
                # Only add visemes with meaningful weight
                if final_weight > 0.05:
                    active_visemes.append({
                        "viseme": v["viseme"],
                        "weight": final_weight
                    })
        
        # Print debugging info occasionally
        if frame_time % 0.5 < 0.02:  # Every 0.5 seconds approximately
            print(f"Frame time {frame_time:.2f}s - Active visemes: {[(v['viseme'], v['weight']) for v in active_visemes]}")
        
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
        
        # Apply each viseme with its weight
        final_landmarks = np.zeros_like(neutral_mouth_landmarks)
        for v in active_visemes:
            viseme_landmarks = self.viseme_generator.apply_viseme(
                neutral_mouth_landmarks, 
                v["viseme"], 
                1.5  # Increased strength for more visible effect
            )
            final_landmarks += viseme_landmarks * v["weight"]
        
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
        
        # Ensure the mouth indices don't exceed the landmark count
        valid_indices = [idx for idx in self.viseme_generator.mouth_indices if idx < len(integrated_landmarks)]
        
        # Replace only the mouth region with the viseme landmarks
        for i, idx in enumerate(valid_indices):
            if i < len(mouth_landmarks):  # Add this check
                # Only update x and y, keep z as is
                integrated_landmarks[idx, 0:2] = mouth_landmarks[i]
        
        # Add debug print to verify modification
        print(f"Applied mouth landmarks to {len(valid_indices)} facial points")
        
        return integrated_landmarks
    
    def animate_from_audio(self, 
                          video_path: str, 
                          audio_path: str, 
                          output_path: str, 
                          emotions: List[Dict] = None, 
                          background_path: str = None,
                          debug: bool = True):
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
        
        # After generating viseme sequence
        viseme_sequence = self.audio_processor.generate_viseme_sequence(audio_path)
        
        # Log which visemes were detected
        viseme_types = set(v["viseme"] for v in viseme_sequence)
        print(f"Detected viseme types: {viseme_types}")

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
        

        # Create debug directory if needed
        debug_dir = None
        if debug:
            debug_dir = os.path.join(os.path.dirname(output_path), "debug_frames")
            os.makedirs(debug_dir, exist_ok=True)
        

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
        
        # Save debug frames
        if debug and frame_index % 10 == 0:  # Save every 10th frame for debugging
            debug_frame = frame.copy()
            
            # Add more visualization
            if face_landmarks:
                # Draw original face landmarks in blue
                mp_drawing = mp.solutions.drawing_utils
                mp_face_mesh = mp.solutions.face_mesh
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
                
                # Draw modified face landmarks in green
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=final_face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
            
            debug_path = os.path.join(debug_dir, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(debug_path, debug_frame)
            
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