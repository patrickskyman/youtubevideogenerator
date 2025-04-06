import os
import cv2
import numpy as np
import torch
import librosa
import subprocess
from tqdm import tqdm

class Wav2LipIntegration:
    """Integration of Wav2Lip pretrained model with FacialAnimationSystem"""
    def __init__(self, checkpoint_path, face_detection_model='face_detection/detection/sfd/'):
        """
        Initialize Wav2Lip integration
        
        Args:
            checkpoint_path: Path to the pretrained Wav2Lip model (.pth file)
            face_detection_model: Path to the face detection model directory
        """
        self.checkpoint_path = checkpoint_path
        self.face_detection_model = face_detection_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Import Wav2Lip modules
        try:
            import sys
            # Add Wav2Lip directory to path
            sys.path.append(os.path.dirname(os.path.dirname(checkpoint_path)))
            
            from models import Wav2Lip
            self.model = Wav2Lip()
            print(f"Loading Wav2Lip model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            s = checkpoint["state_dict"]
            self.model.load_state_dict(s)
            self.model.to(self.device)
            self.model.eval()
            print("Wav2Lip model loaded successfully")
            
            # Import face detector
            if 'sfd' in face_detection_model.lower():
                from face_detection import FaceAlignment, LandmarksType
                self.face_detector = FaceAlignment(LandmarksType._2D, 
                                            flip_input=False, 
                                            device=self.device)
            print("Face detector loaded successfully")
            
        except Exception as e:
            print(f"Failed to load Wav2Lip model: {e}")
            print("Make sure you have the Wav2Lip repository cloned and its dependencies installed")
            self.model = None
    
    def _process_frames(self, frames, mel_chunks):
        """Process video frames with audio chunks using Wav2Lip"""
        # Resize and normalize frames
        img_size = 96  # Wav2Lip uses 96x96 resolution
        frame_h, frame_w = frames[0].shape[:-1]
        
        # Detect faces in first frame
        bbox = self.face_detector.get_detections_for_batch(np.array([frames[0]]))[0]
        if len(bbox) == 0:
            print("No face detected!")
            return None
        
        # Select the largest face
        if len(bbox) > 1:
            bbox = bbox[np.argmax([rect[2] * rect[3] for rect in bbox])]
        else:
            bbox = bbox[0]
        
        x1, y1, x2, y2 = bbox
        roi_size = max(x2 - x1, y2 - y1)
        # Add margin
        x1 = max(0, x1 - roi_size // 10)
        y1 = max(0, y1 - roi_size // 10)
        x2 = min(frame_w, x2 + roi_size // 10)
        y2 = min(frame_h, y2 + roi_size // 10)
        
        # Process each frame
        result_frames = []
        for i, frame in enumerate(frames):
            # Extract face ROI
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (img_size, img_size))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = np.transpose(roi, (2, 0, 1))
            roi = torch.FloatTensor(roi).to(self.device)
            
            # Get corresponding mel chunk
            mel_chunk = torch.FloatTensor(mel_chunks[min(i, len(mel_chunks) - 1)]).to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = self.model(roi.unsqueeze(0), mel_chunk.unsqueeze(0))
                pred = pred[0].cpu().numpy().transpose(1, 2, 0) * 255.0
                pred = pred.astype(np.uint8)
                pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            
            # Resize and paste back
            pred = cv2.resize(pred, (x2 - x1, y2 - y1))
            result_frame = frame.copy()
            result_frame[y1:y2, x1:x2] = pred
            
            result_frames.append(result_frame)
        
        return result_frames
    
    def _extract_mel_features(self, audio_path, fps=25):
        """Extract mel spectrogram features for Wav2Lip"""
        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        mel_step_size = 16
        mel_idx_multiplier = 80./16.
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=800, 
                                           hop_length=200, n_mels=80)
        mel = np.log(mel + 1e-8)
        
        # Prepare chunks
        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                break
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
            i += 1
        
        return mel_chunks
    
    def animate_with_wav2lip(self, video_path, audio_path, output_path):
        """
        Generate a lip-synced video using Wav2Lip
        
        Args:
            video_path: Path to input video with face
            audio_path: Path to audio file for lip sync
            output_path: Path for output video
        """
        if self.model is None:
            print("Wav2Lip model not loaded. Cannot animate.")
            return False
        
        print(f"Processing video: {video_path}")
        print(f"With audio: {audio_path}")
        
        # Extract audio features
        mel_chunks = self._extract_mel_features(audio_path)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_w, frame_h))
        
        # Process frames in batches
        batch_size = 64  # Process this many frames at once for efficiency
        frames = []
        
        with tqdm(total=total_frames, desc="Animating with Wav2Lip") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
                pbar.update(1)
                
                # Process batch when full or at end of video
                if len(frames) >= batch_size or not ret:
                    # Process frames with Wav2Lip
                    result_frames = self._process_frames(frames, mel_chunks)
                    
                    if result_frames:
                        # Write frames to output video
                        for f in result_frames:
                            out.write(f)
                    else:
                        # If processing failed, write original frames
                        for f in frames:
                            out.write(f)
                    
                    # Clear batch
                    frames = []
        
        # Release resources
        cap.release()
        out.release()
        
        # Add audio to the final video
        self._add_audio_to_video(temp_video_path, audio_path, output_path)
        
        # Remove temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        print(f"Video created successfully at {output_path}")
        return True
    
    def _add_audio_to_video(self, video_path, audio_path, output_path):
        """Add audio to video using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            print(f"Successfully added audio to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio: {e}")
            return False