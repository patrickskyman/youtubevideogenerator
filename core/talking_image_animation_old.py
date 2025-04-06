import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import subprocess
import os
from pydub import AudioSegment
import tempfile

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=80, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length

    def _convert_audio_to_wav(self, audio_path: str) -> str:
        """
        Convert input audio file to WAV format using ffmpeg.
        Returns the path to the converted WAV file.
        """
        import os
        
        # Generate a temporary output path for the WAV file
        output_dir = os.path.dirname(audio_path) or "."
        output_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".wav"
        wav_path = os.path.join(output_dir, output_filename)

        try:
            # Use ffmpeg to convert the audio to WAV
            command = [
                "ffmpeg",
                "-i", audio_path,  # Input file
                "-acodec", "pcm_s16le",  # Linear PCM, 16-bit, little-endian
                "-ac", "1",  # Mono channel
                "-ar", str(self.sample_rate),  # Set sample rate
                wav_path,  # Output file
                "-y"  # Overwrite output file if it exists
            ]

            # Run the ffmpeg command and capture output
            process = subprocess.run(command, capture_output=True, text=True, check=True)

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {process.stderr}")

            print(f"Successfully converted {audio_path} to {wav_path}")
            return wav_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error converting audio with FFmpeg: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during audio conversion: {str(e)}")

    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract mel-spectrogram features from audio file"""
        # Convert to WAV if needed
        wav_path = self._convert_audio_to_wav(audio_path)
        
        try:
            # Load audio
            audio, _ = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            normalized = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
            
            return normalized
        finally:
            # Clean up temporary file if created
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        # Convert to WAV if needed
        wav_path = self._convert_audio_to_wav(audio_path)
        try:
            # Try using librosa with error handling for both potential exception types
            try:
                audio, sr = librosa.load(wav_path, sr=None)
                return librosa.get_duration(y=audio, sr=sr)
            except Exception as e:
                # Fallback method using pydub if available
                from pydub import AudioSegment
                audio = AudioSegment.from_file(wav_path)
                return len(audio) / 1000.0  # pydub duration is in milliseconds
        finally:
            # Clean up temporary file if created
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)
    
    def generate_viseme_sequence(self, audio_path: str) -> List[Dict[str, float]]:
        """
        Generate sequence of visemes (visual phonemes) from audio
        Returns a list of dictionaries with viseme and timestamp
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Extract phoneme timing using librosa onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=sr,
            hop_length=self.hop_length,
            units='time'
        )
        
        # Define basic visemes (mouth shapes)
        visemes = ["rest", "open", "wide_open", "rounded", "closed", "slight_open"]
        
        # For production, this would use a trained model to predict visemes
        # Here we'll use a simplified approach for demonstration
        viseme_sequence = []
        
        # Generate viseme for each detected onset
        for i, onset_time in enumerate(onset_frames):
            # Duration until next onset or fixed duration for last onset
            duration = onset_frames[i+1] - onset_time if i < len(onset_frames) - 1 else 0.2
            
            # For demo, select viseme based on audio intensity around onset
            audio_segment = audio[int(onset_time * sr):int((onset_time + 0.1) * sr)]
            intensity = np.abs(audio_segment).mean()
            
            # Map intensity to viseme (simplified approach)
            viseme_idx = min(int(intensity * 20), len(visemes) - 1)
            viseme = visemes[viseme_idx]
            
            viseme_sequence.append({
                "viseme": viseme,
                "start_time": onset_time,
                "duration": duration,
                "intensity": float(intensity)
            })
        
        return viseme_sequence


class LipSyncNetwork(nn.Module):
    """Neural network for mapping audio features to facial landmarks"""
    def __init__(self, audio_features=80, hidden_dim=256, landmark_points=468):
        super().__init__()
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Output mouth landmark displacements (only mouth region landmarks)
        # MediaPipe Face Mesh indices for mouth region: approximately 20 points
        self.mouth_landmark_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20 * 2)  # x, y coordinates for 20 mouth landmarks
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            audio_features: Tensor of shape [batch_size, sequence_length, audio_features]
        Returns:
            Tensor of shape [batch_size, sequence_length, 20*2] (mouth landmark displacements)
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # Process each time step
        mouth_movements = []
        for t in range(seq_len):
            # Encode audio features
            audio_encoded = self.audio_encoder(audio_features[:, t])
            mouth_movements.append(audio_encoded)
        
        # Stack time steps
        mouth_movements = torch.stack(mouth_movements, dim=1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(mouth_movements)
        
        # Predict mouth landmark displacements
        mouth_landmarks = self.mouth_landmark_predictor(lstm_out)
        
        return mouth_landmarks


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
        
        # Indices for mouth landmarks in MediaPipe Face Mesh
        # These are approximate, refer to MediaPipe documentation for exact indices
        self.mouth_indices = [
            0, 11, 12, 13, 14, 15, 16, 17,  # Outer lip
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47  # Inner lip
        ]
        
        # Predefined viseme configurations (displacements from neutral position)
        # In production, these would be learned from data
        self.viseme_configs = {
            "rest": np.zeros((len(self.mouth_indices), 2)),
            "open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.05], [0, -0.07], [0, -0.08], [0, -0.08], [0, -0.07], [0, -0.05], [0, 0],
                [0, 0.05], [0, 0.07], [0, 0.08], [0, 0.07]
            ]),
            "wide_open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.1], [0, -0.12], [0, -0.15], [0, -0.15], [0, -0.12], [0, -0.1], [0, 0],
                [0, 0.1], [0, 0.12], [0, 0.15], [0, 0.12]
            ]),
            "rounded": np.array([
                [0.03, 0], [0.04, 0], [0.04, 0], [0.03, 0], [-0.03, 0], [-0.04, 0], [-0.04, 0], [-0.03, 0],
                [0.02, -0.02], [0.03, -0.03], [0, -0.04], [-0.03, -0.03], [-0.02, -0.02],
                [0.02, 0.02], [0.03, 0.03], [0, 0.04], [-0.03, 0.03], [-0.02, 0.02]
            ]),
            "closed": np.zeros((len(self.mouth_indices), 2)),
            "slight_open": np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [0, -0.03], [0, -0.04], [0, -0.05], [0, -0.05], [0, -0.04], [0, -0.03], [0, 0],
                [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.04]
            ])
        }
        
    def extract_neutral_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Extract neutral facial landmarks from an image"""
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image")
        
        # Extract mouth landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get mouth landmarks (normalized coordinates)
        mouth_landmarks = np.array([
            [face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
            for idx in self.mouth_indices
        ])
        
        return mouth_landmarks
    
    def generate_landmark_sequence(self, 
                                neutral_landmarks: np.ndarray,
                                viseme_sequence: List[Dict[str, float]],
                                fps: int = 30) -> List[np.ndarray]:
        """
        Generate a sequence of landmarks based on viseme sequence
        """
        # Calculate total duration
        if not viseme_sequence:
            return [neutral_landmarks]
            
        total_duration = viseme_sequence[-1]["start_time"] + viseme_sequence[-1]["duration"]
        total_frames = int(total_duration * fps)
        
        landmark_sequence = []
        
        current_viseme_idx = 0
        current_viseme = viseme_sequence[0]
        
        # Ensure all viseme configs have the same shape as neutral_landmarks
        for viseme_name in self.viseme_configs:
            if self.viseme_configs[viseme_name].shape != (len(neutral_landmarks), 2):
                print(f"Fixing shape of viseme '{viseme_name}' from {self.viseme_configs[viseme_name].shape} to {(len(neutral_landmarks), 2)}")
                # Create proper sized array and copy values or pad with zeros
                new_config = np.zeros((len(neutral_landmarks), 2))
                min_rows = min(self.viseme_configs[viseme_name].shape[0], len(neutral_landmarks))
                new_config[:min_rows] = self.viseme_configs[viseme_name][:min_rows]
                self.viseme_configs[viseme_name] = new_config
        
        for frame_idx in range(total_frames):
            # Rest of the method remains the same
            frame_time = frame_idx / fps
            
            # Find current viseme
            while (current_viseme_idx < len(viseme_sequence) - 1 and 
                frame_time > viseme_sequence[current_viseme_idx]["start_time"] + 
                viseme_sequence[current_viseme_idx]["duration"]):
                current_viseme_idx += 1
                current_viseme = viseme_sequence[current_viseme_idx]
            
            # Calculate blend factor for smooth transitions
            next_viseme_idx = min(current_viseme_idx + 1, len(viseme_sequence) - 1)
            next_viseme = viseme_sequence[next_viseme_idx]
            
            # Time within current viseme
            rel_time = frame_time - current_viseme["start_time"]
            viseme_duration = current_viseme["duration"]
            
            # Blend between visemes for smooth transitions
            if rel_time < 0:
                # Before first viseme, use neutral
                blend_factor = 0
                current_config = self.viseme_configs["rest"]
                next_config = self.viseme_configs[current_viseme["viseme"]]
            elif rel_time > viseme_duration and current_viseme_idx < len(viseme_sequence) - 1:
                # Transition to next viseme
                transition_duration = 0.05  # 50ms transition
                rel_transition_time = rel_time - viseme_duration
                blend_factor = min(1.0, rel_transition_time / transition_duration)
                current_config = self.viseme_configs[current_viseme["viseme"]]
                next_config = self.viseme_configs[next_viseme["viseme"]]
            else:
                # Within current viseme - adjust intensity over time for natural movement
                # Ramp up quickly, hold, then ramp down
                if rel_time < 0.05:
                    intensity = rel_time / 0.05
                elif rel_time > viseme_duration - 0.05:
                    intensity = (viseme_duration - rel_time) / 0.05
                else:
                    intensity = 1.0
                    
                intensity *= current_viseme["intensity"]
                blend_factor = 0
                current_config = self.viseme_configs["rest"]
                next_config = self.viseme_configs[current_viseme["viseme"]] * intensity
            
            # Blend configurations
            blended_config = (1 - blend_factor) * current_config + blend_factor * next_config
            
            # Apply to neutral landmarks
            frame_landmarks = neutral_landmarks + blended_config
            
            landmark_sequence.append(frame_landmarks)
            
        return landmark_sequence


class HeadPoseAnimator:
    """Generate natural head movements to accompany speech"""
    def __init__(self):
        # Parameters for head movement
        self.nod_frequency = 0.5  # Hz
        self.shake_frequency = 0.3  # Hz
        self.tilt_frequency = 0.2  # Hz
        self.movement_scale = 0.02  # Scale of movements
        
    def generate_head_movements(self, duration: float, fps: int = 30) -> List[np.ndarray]:
        """
        Generate natural head movements for the given duration
        
        Args:
            duration: Duration in seconds
            fps: Frames per second
            
        Returns:
            List of transformation matrices for each frame
        """
        total_frames = int(duration * fps)
        movements = []
        
        # Generate semi-random natural head movements
        for frame_idx in range(total_frames):
            time = frame_idx / fps
            
            # Add slight nodding (up/down)
            nod = self.movement_scale * np.sin(2 * np.pi * self.nod_frequency * time)
            
            # Add slight head shake (left/right)
            shake = self.movement_scale * 0.7 * np.sin(2 * np.pi * self.shake_frequency * time + 0.5)
            
            # Add slight tilting
            tilt = self.movement_scale * 0.5 * np.sin(2 * np.pi * self.tilt_frequency * time + 1.0)
            
            # Create transformation matrix (simplified)
            # In a real implementation, this would be a proper 3D transformation
            transform = np.array([
                [1.0, tilt, shake],
                [-tilt, 1.0, nod],
                [0.0, 0.0, 1.0]
            ])
            
            movements.append(transform)
            
        return movements


class TalkingImageAnimator:
    """Main class for animating a static image to match speech"""
    def __init__(self, config=None):
        self.audio_processor = AudioProcessor()
        self.viseme_generator = VisemeGenerator()
        self.head_pose_animator = HeadPoseAnimator()
        
        # Configuration
        self.fps = 30
        self.config = config
        
        # Face mesh for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Optional LipSync model (can be added in later versions)
        self.lip_sync_model = None
    
    def extract_face_landmarks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all facial landmarks and separate into regions"""
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image")
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract all landmarks as normalized coordinates
        all_landmarks = np.array([
            [landmark.x, landmark.y, landmark.z]
            for landmark in face_landmarks.landmark
        ])
        
        # Define landmark indices for different face regions
        # These are approximate, refer to MediaPipe documentation for exact indices
        regions = {
            "mouth": list(range(0, 17)) + list(range(61, 68)),  # Outer and inner lips
            "eyes": list(range(36, 48)),  # Both eyes
            "eyebrows": list(range(17, 27)),  # Both eyebrows
            "nose": list(range(27, 36)),  # Nose
            "jaw": list(range(0, 17)),  # Jaw line
            "face_oval": list(range(0, 17))  # Face outline
        }
        
        # Extract landmarks by region
        landmarks_by_region = {}
        for region_name, indices in regions.items():
            valid_indices = [i for i in indices if i < len(all_landmarks)]
            if valid_indices:
                landmarks_by_region[region_name] = all_landmarks[valid_indices]
        
        # Add complete set
        landmarks_by_region["all"] = all_landmarks
        
        return landmarks_by_region
    
    def animate_image(self, 
                     source_image: np.ndarray, 
                     audio_path: str, 
                     output_path: str,
                     background_image: Optional[np.ndarray] = None,
                     add_natural_movements: bool = True) -> bool:
        """
        Animate a static image to match the speech in the audio file
        
        Args:
            source_image: The source image containing a face
            audio_path: Path to the audio file
            output_path: Path to save the output video
            background_image: Optional background image
            add_natural_movements: Whether to add natural head movements
            
        Returns:
            True if animation was successful, False otherwise
        """
        try:
            print("Starting talking image animation process...")
            
            # Step 1: Extract facial landmarks from source image
            print("Extracting facial landmarks from source image...")
            face_landmarks = self.extract_face_landmarks(source_image)
            neutral_mouth_landmarks = self.viseme_generator.extract_neutral_landmarks(source_image)
            
            # Step 2: Process audio file
            print("Processing audio file...")
            audio_duration = self.audio_processor.get_audio_duration(audio_path)
            viseme_sequence = self.audio_processor.generate_viseme_sequence(audio_path)
            
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"Generated {len(viseme_sequence)} visemes")
            
            # Step 3: Generate mouth movements
            print("Generating mouth movements...")
            mouth_landmarks_sequence = self.viseme_generator.generate_landmark_sequence(
                neutral_mouth_landmarks,
                viseme_sequence,
                fps=self.fps
            )
            
            # Step 4: Generate head movements if requested
            head_transforms = None
            if add_natural_movements:
                print("Generating natural head movements...")
                head_transforms = self.head_pose_animator.generate_head_movements(
                    audio_duration,
                    fps=self.fps
                )
            
            # Step 5: Generate frames
            print("Generating animation frames...")
            frames = self._generate_frames(
                source_image,
                mouth_landmarks_sequence,
                head_transforms,
                background_image
            )
            
            # Step 6: Create video with audio
            print("Creating final video with audio...")
            self._create_video_with_audio(frames, audio_path, output_path)
            
            print("Animation completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during animation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_frames(self,
                       source_image: np.ndarray,
                       mouth_landmarks_sequence: List[np.ndarray],
                       head_transforms: Optional[List[np.ndarray]] = None,
                       background_image: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Generate animation frames based on landmark sequences and transforms
        
        Args:
            source_image: Source image
            mouth_landmarks_sequence: Sequence of mouth landmarks for each frame
            head_transforms: Optional sequence of head transformation matrices
            background_image: Optional background image
            
        Returns:
            List of generated frames
        """
        # Get image dimensions
        h, w = source_image.shape[:2]
        frames = []
        
        # Create face mesh for tracking
        face_mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Get source face landmarks for warping
        source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        source_results = face_mesh_detector.process(source_rgb)
        
        if not source_results.multi_face_landmarks:
            raise ValueError("No face detected in the source image")
        
        # Extract source face mesh as numpy array
        source_face_mesh = np.array([
            [landmark.x * w, landmark.y * h]
            for landmark in source_results.multi_face_landmarks[0].landmark
        ], dtype=np.float32)
        
        # Process each frame
        for frame_idx, mouth_landmarks in enumerate(mouth_landmarks_sequence):
            # Start with a copy of the source image
            frame = source_image.copy()
            
            # Convert normalized mouth landmarks to pixel coordinates
            mouth_landmarks_px = np.array([
                [x * w, y * h]
                for x, y in mouth_landmarks
            ], dtype=np.float32)
            
            # Apply head transformation if provided
            if head_transforms and frame_idx < len(head_transforms):
                transform = head_transforms[frame_idx]
                # Apply simplified head transform (in a real implementation, this would be more complex)
                frame = self._apply_head_transform(frame, transform)
            
            # Apply mouth shape deformation
            # For this demonstration, we'll use a simplified approach
            # In a production system, this would be more sophisticated
            # using proper face reenactment techniques
            frame = self._apply_mouth_shape(frame, source_face_mesh, mouth_landmarks_px)
            
            # Apply background if provided
            if background_image is not None:
                # Resize background to match frame size
                bg_resized = cv2.resize(background_image, (w, h))
                
                # Create a face mask
                face_mask = self._create_face_mask(frame, source_results.multi_face_landmarks[0])
                
                # Blend frame with background
                frame = self._blend_with_background(frame, bg_resized, face_mask)
            
            frames.append(frame)
            
            # Print progress
            if frame_idx % 30 == 0:
                print(f"Generated {frame_idx}/{len(mouth_landmarks_sequence)} frames")
        
        return frames
    
    def _apply_head_transform(self, image: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply head transformation to the image"""
        h, w = image.shape[:2]
        
        # Convert to affine transformation matrix for OpenCV
        affine_transform = transform[:2, :].copy()
        
        # Apply transformation
        result = cv2.warpAffine(
            image,
            affine_transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return result
    
    def _apply_mouth_shape(self, image, face_mesh, mouth_landmarks):
        """Apply realistic mouth shape deformation"""
        h, w = image.shape[:2]
        
        # Create a mask for the mouth region
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert landmarks to pixel coordinates
        mouth_points = []
        for x, y in mouth_landmarks:
            mouth_points.append([int(x), int(y)])
        
        # Fill the mouth region
        mouth_points = np.array(mouth_points, dtype=np.int32)
        cv2.fillConvexPoly(mouth_mask, mouth_points, 255)
        
        # Apply Gaussian blur to create feathered edges
        mouth_mask = cv2.GaussianBlur(mouth_mask, (11, 11), 5)
        
        # Create a target image with the new mouth shape
        target_image = image.copy()
        
        # Apply Delaunay triangulation for better warping
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        
        for point in face_mesh:
            subdiv.insert((int(point[0]), int(point[1])))
        
        # Get triangles and apply affine transformation for each
        triangles = subdiv.getTriangleList()
        
        # For each triangle
        for triangle in triangles:
            x1, y1, x2, y2, x3, y3 = triangle
            
            # Check if triangle is within mouth region
            if (cv2.pointPolygonTest(mouth_points, (x1, y1), False) >= 0 or
                cv2.pointPolygonTest(mouth_points, (x2, y2), False) >= 0 or
                cv2.pointPolygonTest(mouth_points, (x3, y3), False) >= 0):
                
                # Find corresponding triangle in the target
                # This is simplified - in practice you'd match vertices
                src_tri = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
                dst_tri = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
                
                # Adjust dst_tri based on mouth_landmarks
                # You would need more sophisticated matching here
                
                # Apply affine transformation
                warp_mat = cv2.getAffineTransform(src_tri[:3], dst_tri[:3])
                warped_triangle = cv2.warpAffine(image, warp_mat, (w, h))
                
                # Mask and blend
                triangle_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(triangle_mask, np.int32([dst_tri]), 255)
                triangle_mask = cv2.bitwise_and(triangle_mask, mouth_mask)
                
                # Blend result
                mask_normalized = triangle_mask.astype(float) / 255.0
                mask_normalized = np.expand_dims(mask_normalized, axis=2)
                
                target_image = target_image * (1 - mask_normalized) + warped_triangle * mask_normalized
        
        return target_image.astype(np.uint8)
        
    def _create_face_mask(self, 
                        image: np.ndarray, 
                        face_landmarks) -> np.ndarray:
        """Create a mask for the face region"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract face contour points
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        face_points = []
        for idx in face_oval_indices:
            try:
                pt = face_landmarks.landmark[idx]
                face_points.append((int(pt.x * w), int(pt.y * h)))
            except IndexError:
                continue
        
        if face_points:
            # Draw filled polygon for face
            face_points = np.array(face_points, dtype=np.int32)
            cv2.fillPoly(mask, [face_points], 255)
            
            # Apply Gaussian blur to feather the edges
            mask = cv2.GaussianBlur(mask, (15, 15), 10)
        
        return mask
    
    def _blend_with_background(self, 
                            foreground: np.ndarray, 
                            background: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """Blend foreground with background using mask"""
        # Normalize mask to range [0, 1]
        mask_norm = mask.astype(np.float32) / 255.0
        mask_norm = mask_norm[..., np.newaxis]
        
        # Blend images
        blended = foreground * mask_norm + background * (1 - mask_norm)
        
        return blended.astype(np.uint8)
    
    def _create_video_with_audio(self, frames: List[np.ndarray], audio_path: str, output_path: str) -> None:
        if not frames:
            raise ValueError("No frames to create video")

        h, w = frames[0].shape[:2]
        
        # Try different codecs available on macOS
        temp_output = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try 'avc1' instead of 'mp4v'
        
        print(f"Attempting to create video at {temp_output} with dimensions {w}x{h}")

        video = cv2.VideoWriter(temp_output, fourcc, self.fps, (w, h))
        if not video.isOpened():
            # If avc1 fails, try another codec
            video.release()
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            video = cv2.VideoWriter(temp_output, fourcc, self.fps, (w, h))
            
        if not video.isOpened():
            # If all codecs fail, try a different container format
            video.release()
            temp_output = output_path + ".temp.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video = cv2.VideoWriter(temp_output, fourcc, self.fps, (w, h))
            
        if not video.isOpened():
            raise RuntimeError(f"Could not create video writer with any codec. Check OpenCV installation.")

        frame_count = 0
        for frame in frames:
            if frame is not None:  # Ensure frame is not None
                video.write(frame)
                frame_count += 1
            else:
                print(f"Warning: Frame {frame_count} is None, skipping.")

        video.release()

        # Check if the file was created
        import os
        if not os.path.exists(temp_output):
            raise FileNotFoundError(f"Video file {temp_output} was not created. Check codec and permissions.")

        print(f"Video saved without audio at {temp_output} with {frame_count} frames.")

        # Use ffmpeg to add audio
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_output,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            "-shortest",
            output_path
        ]

        print(f"Running FFmpeg command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            os.remove(temp_output)
            print("FFmpeg completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio to video: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            print(f"FFmpeg stdout: {e.stdout}")
            # If ffmpeg fails, at least keep the video without audio
            if os.path.exists(temp_output):
                os.rename(temp_output, output_path)
                print(f"Kept video without audio at {output_path}")

def main():
    """Main function to demonstrate the TalkingImageAnimator"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Talking Image Animation")
    parser.add_argument("--source_image", required=True, help="Path to source image")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--background", help="Optional path to background image")
    parser.add_argument("--no_head_movement", action="store_true", help="Disable natural head movements")
    
    args = parser.parse_args()
    
    # Load source image
    source_image = cv2.imread(args.source_image)
    if source_image is None:
        print(f"Error: Could not load source image from {args.source_image}")
        return
    
    # Load background image if provided
    background_image = None
    if args.background:
        background_image = cv2.imread(args.background)
        if background_image is None:
            print(f"Warning: Could not load background image from {args.background}")
    
    # Create animator
    animator = TalkingImageAnimator()
    
    # Animate image
    success = animator.animate_image(
        source_image=source_image,
        audio_path=args.audio,
        output_path=args.output,
        background_image=background_image,
        add_natural_movements=not args.no_head_movement
    )
    
    if success:
        print(f"Animation completed successfully! Output saved to {args.output}")
    else:
        print("Animation failed.")


class LipSyncIntegrator:
    """Integrate lip sync animation with the existing video generation architecture"""
    def __init__(self):
        self.talking_animator = TalkingImageAnimator()
        
    def integrate_with_generator(self, 
                               source_image: np.ndarray,
                               audio_path: str,
                               generator: object) -> List[np.ndarray]:
        """
        Integrate lip sync animation with the video generator
        
        Args:
            source_image: Source face image
            audio_path: Path to audio file
            generator: Instance of AdvancedVideoGenerator
            
        Returns:
            List of animated frames
        """
        # Extract audio features
        audio_processor = AudioProcessor()
        viseme_sequence = audio_processor.generate_viseme_sequence(audio_path)
        audio_duration = audio_processor.get_audio_duration(audio_path)
        
        # Extract facial landmarks
        face_landmarks = self.talking_animator.extract_face_landmarks(source_image)
        neutral_mouth_landmarks = self.talking_animator.viseme_generator.extract_neutral_landmarks(source_image)
        
        # Generate mouth movements
        mouth_landmarks_sequence = self.talking_animator.viseme_generator.generate_landmark_sequence(
            neutral_mouth_landmarks,
            viseme_sequence,
            fps=self.talking_animator.fps
        )
        
        # Generate head movements
        head_transforms = self.talking_animator.head_pose_animator.generate_head_movements(
            audio_duration,
            fps=self.talking_animator.fps
        )
        
        # Generate frames
        frames = self.talking_animator._generate_frames(
            source_image,
            mouth_landmarks_sequence,
            head_transforms
        )
        
        # If generator is provided, we can further process frames
        if generator:
            enhanced_frames = []
            for frame in frames:
                # Process through generator pipeline
                enhanced_frame = generator.process_single_frame(frame)
                enhanced_frames.append(enhanced_frame)
            return enhanced_frames
        
        return frames


class EnhancedTalkingHeadModel:
    """Enhanced model for realistic talking head animation with emotion control"""
    def __init__(self):
        self.animator = TalkingImageAnimator()
        self.audio_processor = AudioProcessor()
        
        # Emotion parameters
        self.emotion_mappings = {
            "neutral": {
                "mouth_scale": 1.0,
                "head_movement_scale": 1.0,
                "blink_rate": 0.15  # Blinks per second
            },
            "happy": {
                "mouth_scale": 1.2,  # Slightly larger mouth movements
                "head_movement_scale": 1.3,  # More animated head movements
                "blink_rate": 0.18  # More frequent blinking
            },
            "sad": {
                "mouth_scale": 0.8,  # Smaller mouth movements
                "head_movement_scale": 0.7,  # Less head movement
                "blink_rate": 0.1  # Less frequent blinking
            },
            "angry": {
                "mouth_scale": 1.1,  # Slightly pronounced mouth movements
                "head_movement_scale": 1.2,  # More intense movements
                "blink_rate": 0.2  # More frequent blinking
            },
            "surprised": {
                "mouth_scale": 1.3,  # Exaggerated mouth movements
                "head_movement_scale": 1.5,  # Exaggerated head movements
                "blink_rate": 0.05  # Very little blinking
            }
        }
        
    def animate_with_emotion(self,
                           source_image: np.ndarray,
                           audio_path: str,
                           output_path: str,
                           emotion: str = "neutral",
                           background_image: Optional[np.ndarray] = None) -> bool:
        """
        Animate the source image with specified emotional style
        
        Args:
            source_image: Source face image
            audio_path: Path to audio file
            output_path: Path to output video
            emotion: Emotional style to apply
            background_image: Optional background image
            
        Returns:
            True if animation was successful
        """
        if emotion not in self.emotion_mappings:
            print(f"Warning: Unknown emotion '{emotion}', defaulting to 'neutral'")
            emotion = "neutral"
            
        emotion_params = self.emotion_mappings[emotion]
        
        # Process audio
        audio_duration = self.audio_processor.get_audio_duration(audio_path)
        viseme_sequence = self.audio_processor.generate_viseme_sequence(audio_path)
        
        # Apply emotion intensity to visemes
        for viseme in viseme_sequence:
            viseme["intensity"] *= emotion_params["mouth_scale"]
        
        # Extract facial landmarks
        face_landmarks = self.animator.extract_face_landmarks(source_image)
        neutral_mouth_landmarks = self.animator.viseme_generator.extract_neutral_landmarks(source_image)
        
        # Generate mouth movements
        mouth_landmarks_sequence = self.animator.viseme_generator.generate_landmark_sequence(
            neutral_mouth_landmarks,
            viseme_sequence,
            fps=self.animator.fps
        )
        
        # Generate head movements with emotion-specific scaling
        self.animator.head_pose_animator.movement_scale *= emotion_params["head_movement_scale"]
        head_transforms = self.animator.head_pose_animator.generate_head_movements(
            audio_duration,
            fps=self.animator.fps
        )
        
        # Generate frames
        frames = self.animator._generate_frames(
            source_image,
            mouth_landmarks_sequence,
            head_transforms,
            background_image
        )
        
        # Add blinking based on emotion parameters
        frames = self._add_blinking(
            frames, 
            audio_duration, 
            self.animator.fps,
            emotion_params["blink_rate"]
        )
        
        # Create video with audio
        self.animator._create_video_with_audio(frames, audio_path, output_path)
        
        return True
    
    def _add_blinking(self, 
                    frames: List[np.ndarray],
                    duration: float,
                    fps: int,
                    blink_rate: float) -> List[np.ndarray]:
        """
        Add natural blinking to animation frames
        
        Args:
            frames: List of animation frames
            duration: Animation duration in seconds
            fps: Frames per second
            blink_rate: Average blinks per second
            
        Returns:
            Frames with blinking added
        """
        num_frames = len(frames)
        
        # Generate blink timings (semi-random)
        num_blinks = int(duration * blink_rate)
        blink_frames = []
        
        for _ in range(num_blinks):
            # Random frame for blink start
            blink_start = np.random.randint(0, num_frames - int(0.15 * fps))
            blink_frames.extend(range(blink_start, blink_start + int(0.15 * fps)))
        
        # Apply blinks to frames
        for frame_idx in blink_frames:
            if 0 <= frame_idx < num_frames:
                # Simplified blink effect - in production this would be more sophisticated
                # Here we just darken a region around the eyes
                frame = frames[frame_idx]
                h, w = frame.shape[:2]
                
                # Get face landmarks to locate eyes
                face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.animator.face_mesh.process(face_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Eye indices (approximate for MediaPipe Face Mesh)
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
                    
                    # Calculate eye regions
                    for eye_indices in [left_eye_indices, right_eye_indices]:
                        eye_points = []
                        for idx in eye_indices:
                            if idx < len(landmarks):
                                pt = landmarks[idx]
                                x, y = int(pt.x * w), int(pt.y * h)
                                eye_points.append([x, y])
                        
                        if eye_points:
                            eye_points = np.array(eye_points, dtype=np.int32)
                            
                            # Create eye mask
                            eye_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(eye_mask, [eye_points], 255)
                            
                            # Apply blink effect
                            blink_factor = 0.7  # Darkening factor
                            frame[eye_mask > 0] = (frame[eye_mask > 0] * blink_factor).astype(np.uint8)
                    
                    frames[frame_idx] = frame
        
        return frames


class VideoWithTalkingImageGenerator:
    """Generate videos with talking image integration for the main video generation pipeline"""
    def __init__(self, config=None):
        self.talking_animator = TalkingImageAnimator()
        self.enhanced_model = EnhancedTalkingHeadModel()
        self.config = config
        
    def generate_video(self, 
                      source_image: np.ndarray,
                      audio_path: str,
                      driving_frames: Optional[List[np.ndarray]] = None,
                      background_image: Optional[np.ndarray] = None,
                      emotion: str = "neutral",
                      output_path: str = "output.mp4") -> str:
        """
        Generate video with talking animation
        
        Args:
            source_image: Source face image
            audio_path: Path to audio file
            driving_frames: Optional list of driving frames (for motion reference)
            background_image: Optional background image
            emotion: Emotional style to apply
            output_path: Path to save output video
            
        Returns:
            Path to generated video
        """
        if driving_frames:
            # If driving frames are provided, integrate talking animation with them
            print("Using driving frames to guide animation...")
            
            # Extract audio features
            audio_processor = AudioProcessor()
            viseme_sequence = audio_processor.generate_viseme_sequence(audio_path)
            audio_duration = audio_processor.get_audio_duration(audio_path)
            
            # Generate talking animation frames
            frames = self._generate_talking_frames_with_driving(
                source_image,
                driving_frames,
                viseme_sequence,
                audio_duration,
                emotion,
                background_image
            )
            
            # Create video with audio
            self.talking_animator._create_video_with_audio(frames, audio_path, output_path)
            
        else:
            # If no driving frames, use direct talking animation
            print("Generating talking animation from static image...")
            
            self.enhanced_model.animate_with_emotion(
                source_image,
                audio_path,
                output_path,
                emotion,
                background_image
            )
            
        return output_path
    
    def _generate_talking_frames_with_driving(self,
                                           source_image: np.ndarray,
                                           driving_frames: List[np.ndarray],
                                           viseme_sequence: List[Dict[str, float]],
                                           audio_duration: float,
                                           emotion: str = "neutral",
                                           background_image: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Generate talking animation frames guided by driving frames
        
        Args:
            source_image: Source face image
            driving_frames: List of driving frames for motion reference
            viseme_sequence: Sequence of visemes from audio
            audio_duration: Audio duration in seconds
            emotion: Emotional style
            background_image: Optional background image
            
        Returns:
            List of generated animation frames
        """
        # Get emotion parameters
        emotion_params = self.enhanced_model.emotion_mappings.get(
            emotion, 
            self.enhanced_model.emotion_mappings["neutral"]
        )
        
        # Extract facial landmarks from source image
        face_landmarks = self.talking_animator.extract_face_landmarks(source_image)
        neutral_mouth_landmarks = self.talking_animator.viseme_generator.extract_neutral_landmarks(source_image)
        
        # Apply emotion intensity to visemes
        for viseme in viseme_sequence:
            viseme["intensity"] *= emotion_params["mouth_scale"]
        
        # Generate mouth landmarks sequence
        mouth_landmarks_sequence = self.talking_animator.viseme_generator.generate_landmark_sequence(
            neutral_mouth_landmarks,
            viseme_sequence,
            fps=self.talking_animator.fps
        )
        
        # Calculate frames needed
        num_frames = min(
            len(driving_frames),
            len(mouth_landmarks_sequence),
            int(audio_duration * self.talking_animator.fps)
        )
        
        # Generate head transforms from driving frames
        head_transforms = self._extract_head_movements_from_driving(
            driving_frames[:num_frames],
            emotion_params["head_movement_scale"]
        )
        
        # Generate frames
        frames = self.talking_animator._generate_frames(
            source_image,
            mouth_landmarks_sequence[:num_frames],
            head_transforms,
            background_image
        )
        
        # Add blinking
        frames = self.enhanced_model._add_blinking(
            frames, 
            audio_duration, 
            self.talking_animator.fps,
            emotion_params["blink_rate"]
        )
        
        return frames
    
    def _extract_head_movements_from_driving(self,
                                          driving_frames: List[np.ndarray],
                                          movement_scale: float = 1.0) -> List[np.ndarray]:
        """
        Extract head movement transforms from driving frames
        
        Args:
            driving_frames: List of driving video frames
            movement_scale: Scale factor for head movements
            
        Returns:
            List of transformation matrices
        """
        transforms = []
        prev_landmarks = None
        
        # Face mesh for tracking
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        for frame in driving_frames:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                # Extract facial landmarks
                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in results.multi_face_landmarks[0].landmark
                ])
                
                if prev_landmarks is not None:
                    # Calculate transformation from previous frame to current
                    # This is a simplified approach - in production, use proper 3D pose estimation
                    
                    # Use rigid landmarks (eyes, nose) to estimate head movement
                    rigid_indices = list(range(27, 51))  # Approximate eye and nose region
                    
                    # Calculate translation
                    translation = np.mean(landmarks[rigid_indices, :2] - prev_landmarks[rigid_indices, :2], axis=0)
                    
                    # Scale by movement_scale
                    translation *= movement_scale
                    
                    # Create affine transformation matrix
                    transform = np.array([
                        [1.0, 0.0, translation[0]],
                        [0.0, 1.0, translation[1]],
                        [0.0, 0.0, 1.0]
                    ])
                else:
                    # First frame - identity transform
                    transform = np.eye(3)
                
                prev_landmarks = landmarks
            else:
                # No face detected - identity transform
                transform = np.eye(3)
            
            transforms.append(transform)
        
        return transforms


# Integration with the main AdvancedVideoGenerator
def integrate_talking_animation_with_generator(config=None):
    """
    Function to integrate talking animation with the main AdvancedVideoGenerator
    
    Args:
        config: Configuration for the video generator
        
    Returns:
        Modified AdvancedVideoGenerator class with talking animation support
    """
    from essentials import AdvancedVideoConfig
    
    # This function would be called to integrate the talking animation functionality
    # with the existing AdvancedVideoGenerator class
    
    # Example integration code:
    def extended_generate_video(self, source_image_path, audio_path, output_path, 
                              background_image_path=None, emotion="neutral", 
                              use_driving_video=False, driving_video_path=None):
        """Extended video generation method with talking animation support"""
        # Load source image
        source_image = cv2.imread(source_image_path)
        if source_image is None:
            raise ValueError(f"Could not load source image from {source_image_path}")
        
        # Load background if provided
        background_image = None
        if background_image_path:
            background_image = cv2.imread(background_image_path)
        
        # Initialize talking image generator
        talking_generator = VideoWithTalkingImageGenerator(self.config)
        
        if use_driving_video and driving_video_path:
            # Load driving video frames
            driving_frames = []
            cap = cv2.VideoCapture(driving_video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                driving_frames.append(frame)
            cap.release()
            
            # Generate video with driving frames
            return talking_generator.generate_video(
                source_image,
                audio_path,
                driving_frames,
                background_image,
                emotion,
                output_path
            )
        else:
            # Generate video from static image
            return talking_generator.generate_video(
                source_image,
                audio_path,
                None,
                background_image,
                emotion,
                output_path
            )
    
    # The integration would patch the AdvancedVideoGenerator class
    # to include talking animation functionality
    
    return extended_generate_video


if __name__ == "__main__":
    main()