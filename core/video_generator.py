import os
import torch
import mediapipe as mp
import numpy as np
import argparse
import cv2
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from PIL import Image
from absl import logging 
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import dlib
from scipy.spatial import Delaunay
import kornia
import torch.nn.functional as F
import torch.nn as nn
# Update AdvancedVideoGenerator class
from essentials import AdvancedVideoConfig, BackgroundProcessor, DenseMotionNetwork, ExpressionTransfer, FaceMeshDetector, FrameInterpolator, KeypointDetector, MultiFaceProcessor, QualityAssessor, QualityMetrics, RealTimeProcessor, StyleTransfer
from essentials import VideoStabilizer
from enhanced_face_swapper import EnhancedFaceSwapper

class AdvancedVideoGenerator(nn.Module):
    """Advanced video generation with face swapping, style transfer, and stabilization"""
    def __init__(self, config: AdvancedVideoConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.quality_metrics = QualityMetrics()
        self.interpolator = FrameInterpolator(config)
        self.multi_face_processor = MultiFaceProcessor(config)
        self.real_time_processor = RealTimeProcessor(config)
        self.face_detector = FaceMeshDetector()
        self.quality_assessor = QualityAssessor()
        self.background_processor = BackgroundProcessor()
        self.kp_detector = KeypointDetector(config)
        self.dense_motion_network = DenseMotionNetwork(config)
        
        # New components
        self.face_swapper = EnhancedFaceSwapper()
        self.style_transfer = StyleTransfer()
        self.expression_transfer = ExpressionTransfer()
        self.video_stabilizer = VideoStabilizer()
        
        # Processing states
        self.current_style_image = None
        self.last_processed_frame = None
        self.frame_buffer = []
        
    def stop_real_time(self):
        """Stop real-time processing"""
        if hasattr(self, 'real_time_processor'):
            self.real_time_processor.stop()

    def forward(self, source_images: Union[torch.Tensor, List[torch.Tensor]],
            driving_frames: List[torch.Tensor],
            background_image: Optional[torch.Tensor] = None,
            style_image: Optional[torch.Tensor] = None,
            expression_intensity: float = 1.0,
            enable_stabilization: bool = True,
            swap_faces: bool = True) -> Dict:
        """
        Generate video with all advanced features.
        
        Args:
            source_images: Source face image(s)
            driving_frames: List of driving video frames
            background_image: Optional background replacement image
            style_image: Optional style reference image
            expression_intensity: Control for expression transfer (0.0 to 1.0)
            enable_stabilization: Whether to enable video stabilization
            swap_faces: Whether to perform face swapping
            
        Returns:
            Dictionary containing processed frames and quality metrics
        """
        # Handle single or multiple source images
        if isinstance(source_images, torch.Tensor):
            source_images = [source_images]
            
        results = {
            'frames': [],
            'quality_scores': [],
            'stability_scores': [],
            'smoothness_scores': [],
            'landmarks': [],
            'expression_scores': [],
            'style_consistency': []
        }
        
        # Initialize style transfer if needed
        if style_image is not None and style_image != self.current_style_image:
            self.current_style_image = style_image
            self._prepare_style_transfer(style_image)
        
        # Process each frame
        prev_frame = None
        for frame_idx, driving_frame in enumerate(driving_frames):
            # Convert tensors to numpy for processing
            driving_np = driving_frame.cpu().numpy()
            source_np = [img.cpu().numpy() for img in source_images]
            
            # Ensure proper shape for processing
            if driving_np.shape[0] == 1:  # Remove batch dimension if present
                driving_np = driving_np[0]
            if driving_np.shape[0] == 3:  # Convert from CHW to HWC
                driving_np = np.transpose(driving_np, (1, 2, 0))
                
            source_np = [np.transpose(img[0], (1, 2, 0)) if img.shape[0] == 1 else 
                        np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img 
                        for img in source_np]
            
            # Convert to uint8 [0-255]
            driving_np = (driving_np * 255).clip(0, 255).astype(np.uint8)
            source_np = [(img * 255).clip(0, 255).astype(np.uint8) for img in source_np]
        
            
            # 1. Face Detection and Landmark Extraction
            print("Driving Frame Shape:", driving_np.shape)
            face_landmarks = self.multi_face_processor.detect_faces(driving_np)
            results['landmarks'].append(face_landmarks)
            
            
            # 2. Face Swapping (if enabled)
            processed_frame = driving_np.copy()
            if swap_faces:
                for face_idx, landmarks in enumerate(face_landmarks):
                    if face_idx < len(source_np):
                        source_img = source_np[face_idx]
                        if len(source_img.shape) == 2:  # Grayscale image
                            source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)
                        elif source_img.shape[2] == 4:  # Image with alpha channel
                            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGRA2BGR)
                        processed_frame = self.face_swapper.swap_faces(
                            source_img,
                            processed_frame
                        )
            
            # Convert back to float32 [0-1] for further processing
            processed_frame = processed_frame.astype(np.float32) / 255.0

            # 3. Expression Transfer
            if expression_intensity > 0:
                for face_idx, landmarks in enumerate(face_landmarks):
                    if face_idx < len(source_np):
                        processed_frame = self.expression_transfer.transfer_expression(
                            source_np[face_idx],
                            processed_frame,
                            expression_intensity
                        )
                        
                # Calculate expression accuracy score
                expr_score = self._calculate_expression_accuracy(
                    source_np[0], processed_frame
                )
                results['expression_scores'].append(expr_score)
            
            # 4. Style Transfer
            if self.current_style_image is not None:
                processed_frame = self._apply_style_transfer(processed_frame)
                style_consistency = self._calculate_style_consistency(
                    processed_frame, self.current_style_image.cpu().numpy()
                )
                results['style_consistency'].append(style_consistency)
            
            # 5. Background Processing
            if background_image is not None:
                foreground, _, mask = self.background_processor.separate_background(
                    processed_frame
                )
                processed_frame = self.background_processor.replace_background(
                    foreground, mask, background_image.cpu().numpy()
                )
            
            # 6. Frame Interpolation
            if prev_frame is not None:
                interpolated = self.interpolator.interpolate(
                    torch.from_numpy(prev_frame).float(),
                    torch.from_numpy(processed_frame).float()
                )
                results['frames'].extend([frame.cpu().numpy() for frame in interpolated])
                print(f"Added {len(interpolated)} interpolated frames")
            results['frames'].append(processed_frame)
            print(f"Total frames after this iteration: {len(results['frames'])}")
                        
            # 7. Video Stabilization
        if enable_stabilization:
            processed_frame_uint8 = (processed_frame * 255).clip(0, 255).astype(np.uint8)
            processed_frame = self.video_stabilizer.stabilize_frame(processed_frame_uint8)
            processed_frame = processed_frame.astype(np.float32) / 255.0
                
            # 8. Quality Assessment
            try:
                processed_tensor = torch.from_numpy(processed_frame).float()
                if len(processed_tensor.shape) == 3:
                    processed_tensor = processed_tensor.permute(2, 0, 1)  # HWC to CHW
                
                quality_score = self.quality_assessor(
                    processed_tensor.unsqueeze(0),  # Add batch dimension
                    driving_frame
                )
            except Exception as e:
                print(f"Warning: Quality assessment failed - {str(e)}")
                quality_score = torch.tensor([0.0])
            
            stability_score = self.quality_metrics.calculate_landmark_stability(
                face_landmarks[0] if face_landmarks else None
            )
            # Get motion flow field if available
            flow_field = getattr(self.dense_motion_network, 'last_flow', None)
            
            # Calculate smoothness score
            smoothness_score = self.quality_metrics.calculate_motion_smoothness(flow_field)
            
            
            
            # Store results
            results['frames'].append(processed_frame)
            results['quality_scores'].append(quality_score)
            results['stability_scores'].append(stability_score)
            results['smoothness_scores'].append(smoothness_score)
            
            prev_frame = processed_frame
            self.last_processed_frame = processed_frame

            # Before adding to results, ensure proper dtype
            if 'frames' in results:
                results['frames'] = [frame.astype(np.float32) / 255.0 if isinstance(frame, np.ndarray) else frame 
                                for frame in results['frames']]
                
        return results
    
    def process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline"""
        # Print frame info for debugging
        print("Driving Frame Shape:", frame.shape)
        
        # Ensure frame is in correct format and size
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        if len(frame.shape) == 4:
            frame = frame.squeeze(0)
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        
        # Resize if needed
        target_size = (256, 256)
        if frame.shape[:2] != target_size:
            frame = cv2.resize(frame, target_size)
        
        # Ensure uint8 type for MediaPipe
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        # Ensure we have a valid source image
        if not hasattr(self, 'source_image') or self.source_image is None:
            print("Error: No source image available")
            return frame
            
        if isinstance(self.source_image, list):
            source_img = self.source_image[0]  # Use first source image
        else:
            source_img = self.source_image
            
        # Convert source image if needed
        if isinstance(source_img, torch.Tensor):
            source_img = source_img.cpu().numpy()
            if source_img.shape[0] == 3:
                source_img = np.transpose(source_img, (1, 2, 0))
            if source_img.max() <= 1.0:
                source_img = (source_img * 255).astype(np.uint8)
        
        # Apply face swapping
        if self.face_swapper is not None:
            try:
                processed_frame = self.face_swapper.swap_faces(source_img, frame)
            except Exception as e:
                print(f"Face swapping failed: {str(e)}")
                processed_frame = frame
        else:
            processed_frame = frame
        
        return processed_frame
    
    def start_real_time(self, 
                       source_images: List[np.ndarray],
                       style_image: Optional[torch.Tensor] = None):
        """Start real-time processing"""
        self.source_image = source_images
        if style_image is not None:
            self.current_style_image = style_image
            self._prepare_style_transfer(style_image)
        self.real_time_processor.start(self)
    
    def _prepare_style_transfer(self, style_image: torch.Tensor):
        """Prepare style transfer model with new style image"""
        self.style_features = self.style_transfer(style_image)
    
    def _apply_style_transfer(self, frame: np.ndarray) -> np.ndarray:
        """Apply style transfer to a frame"""
        frame_tensor = torch.from_numpy(frame).float()
        styled_frame = self.style_transfer.transfer_style(
            frame_tensor,
            self.current_style_image
        )
        return styled_frame.cpu().numpy()
    
    def _calculate_expression_accuracy(self, 
                                    source_frame: np.ndarray,
                                    generated_frame: np.ndarray) -> float:
        """Calculate how well expressions were transferred"""
        source_expr = self.expression_transfer.extract_expression(source_frame)
        gen_expr = self.expression_transfer.extract_expression(generated_frame)
        
        # Check if we have valid face landmarks
        if (not source_expr.multi_face_landmarks or 
            not gen_expr.multi_face_landmarks):
            return 0.0
        
        # Get landmarks from first face
        source_landmarks = source_expr.multi_face_landmarks[0].landmark
        gen_landmarks = gen_expr.multi_face_landmarks[0].landmark
        
        # Calculate differences in key expression features
        expr_diff = 0.0
        
        # Compare mouth landmarks (indices 0-17)
        mouth_indices = range(0, 17)
        for idx in mouth_indices:
            s_pt = source_landmarks[idx]
            g_pt = gen_landmarks[idx]
            expr_diff += ((s_pt.x - g_pt.x)**2 + 
                        (s_pt.y - g_pt.y)**2 + 
                        (s_pt.z - g_pt.z)**2)
        
        # Compare eye landmarks (indices 33-46 for left eye, 362-374 for right eye)
        eye_indices = list(range(33, 46)) + list(range(362, 374))
        for idx in eye_indices:
            s_pt = source_landmarks[idx]
            g_pt = gen_landmarks[idx]
            expr_diff += ((s_pt.x - g_pt.x)**2 + 
                        (s_pt.y - g_pt.y)**2 + 
                        (s_pt.z - g_pt.z)**2)
        
        # Compare eyebrow landmarks (indices 46-55 for right eyebrow, 285-295 for left eyebrow)
        brow_indices = list(range(46, 55)) + list(range(285, 295))
        for idx in brow_indices:
            s_pt = source_landmarks[idx]
            g_pt = gen_landmarks[idx]
            expr_diff += ((s_pt.x - g_pt.x)**2 + 
                        (s_pt.y - g_pt.y)**2 + 
                        (s_pt.z - g_pt.z)**2)
        
        # Normalize by number of points compared
        total_points = len(mouth_indices) + len(eye_indices) + len(brow_indices)
        expr_diff /= total_points
        
        # Convert to similarity score (1 / (1 + diff))
        return 1.0 / (1.0 + expr_diff)
    
    def _calculate_style_consistency(self,
                                   generated_frame: np.ndarray,
                                   style_frame: np.ndarray) -> float:
        """Calculate style transfer consistency"""
        gen_features = self.style_transfer(
            torch.from_numpy(generated_frame).float()
        )
        style_features = self.style_transfer(
            torch.from_numpy(style_frame).float()
        )
        
        # Calculate style consistency using Gram matrices
        consistency_scores = []
        for gf, sf in zip(gen_features, style_features):
            gen_gram = self.style_transfer._gram_matrix(gf)
            style_gram = self.style_transfer._gram_matrix(sf)
            score = 1.0 / (1.0 + F.mse_loss(gen_gram, style_gram).item())
            consistency_scores.append(score)
            
        return np.mean(consistency_scores)
    
    def _prepare_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image for the video generation pipeline.
        
        Args:
            image_path: Path to the source image
            
        Returns:
            Preprocessed image as a tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to target dimensions
        image = cv2.resize(image, self.config.image_size)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor

    def _prepare_video(self, video_path: str, max_frames: Optional[int] = None) -> List[torch.Tensor]:
        """
        Load and prepare video frames for processing.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (None for all frames)
            
        Returns:
            List of prepared video frames as torch tensors
        """
        print(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {frame_count}")
        
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        
        frames = []
        valid_frames = 0
        
        # Process frames
        for i in range(frame_count):
            print(f"Processing frame {i}/{frame_count}")
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {i}")
                continue
            
            try:
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply face detection
                landmarks = self.face_detector.detect_landmarks(frame_rgb)
                
                # Defensive check for landmarks
                if landmarks is None:
                    print(f"Warning: No face detected in frame {i}")
                    continue
                
                # Check landmark format and structure
                if np.isscalar(landmarks):
                    print(f"Warning: Landmarks is a scalar value in frame {i}")
                    continue
                    
                if not isinstance(landmarks, (list, np.ndarray)) or len(landmarks) == 0:
                    print(f"Warning: No landmarks found in frame {i}")
                    continue
                
                # Save a debug frame with landmarks drawn
                if i == 0 or i == min(5, frame_count-1):
                    debug_frame = frame_rgb.copy()
                    
                    # Handle different landmark formats
                    if isinstance(landmarks, np.ndarray) and landmarks.ndim > 1:
                        # For landmark arrays with shape [n_points, 2+]
                        for lm in landmarks:
                            if len(lm) >= 2:
                                x, y = int(lm[0]), int(lm[1])
                                cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
                    elif isinstance(landmarks, list):
                        # For list of landmarks
                        for lm in landmarks:
                            if hasattr(lm, '__len__') and len(lm) >= 2:
                                x, y = int(lm[0]), int(lm[1])
                                cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
                    
                    debug_path = f"debug_driving_frame_{i}.jpg"
                    cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
                    print(f"Saved debug frame to {debug_path}")
                
                # Convert frame to tensor format suitable for the model
                # Simple preprocessing without relying on config.frame_size
                frame_tensor = self._preprocess_frame_simple(frame_rgb)
                frames.append(frame_tensor)
                valid_frames += 1
                
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Close video file
        cap.release()
        
        # Check if we have valid frames
        if valid_frames == 0:
            raise ValueError("No valid frames with faces were found in the video")
            
        print(f"Successfully processed {valid_frames} frames with detected faces")
        return frames

    def _preprocess_frame_simple(self, frame: np.ndarray) -> torch.Tensor:
        """
        Simplified preprocessing of a video frame for the model,
        without relying on config settings.
        
        Args:
            frame: Input frame as numpy array (RGB format)
            
        Returns:
            Preprocessed frame as torch tensor
        """
        # Convert to tensor and normalize to 0-1 range
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        
        # Change from HWC to CHW format (height, width, channels) -> (channels, height, width)
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor

    def generate_video(self,
        source_images: Union[torch.Tensor, List[torch.Tensor]],
        driving_frames: List[torch.Tensor],
        background_image: Optional[torch.Tensor] = None,
        style_image: Optional[torch.Tensor] = None,
        expression_intensity: float = 1.0,
        enable_stabilization: bool = True,
        swap_faces: bool = True
    ) -> List[np.ndarray]:
        print(f"Number of driving frames: {len(driving_frames)}")
        """
        Generate a video using the AdvancedVideoGenerator model.
        
        Args:
            model: The AdvancedVideoGenerator model
            source_images: Source face images (single or multiple)
            driving_frames: Driving video frames
            background_image: Optional background replacement image
            style_image: Optional style reference image
            expression_intensity: Control for expression transfer (0.0 to 1.0)
            enable_stabilization: Whether to enable video stabilization
            swap_faces: Whether to perform face swapping
            
        Returns:
            List of generated video frames as numpy arrays
        """
        # Process frames through the model
        with torch.no_grad():
            results = self(
                source_images=source_images,
                driving_frames=driving_frames,
                background_image=background_image,
                style_image=style_image,
                expression_intensity=expression_intensity,
                enable_stabilization=enable_stabilization,
                swap_faces=swap_faces
            )
        print(f"Number of generated frames: {len(results['frames'])}")
        
        # Convert from torch tensors to numpy arrays
        frames = results['frames']
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu().numpy() for frame in frames]
        
        # Convert from [0, 1] float to [0, 255] uint8
        frames_uint8 = []
        for frame in frames:
            # Convert from CHW to HWC if needed
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            
            # Ensure frame is in RGB format
            if frame.shape[2] == 3:
                # Convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Convert to uint8
            frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
            frames_uint8.append(frame_uint8)
        
        return frames_uint8

    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
        """Save frames as a video file with proper encoding."""
        if not frames:
            raise ValueError("No frames to save")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Get frame dimensions from the first frame
        height, width = frames[0].shape[:2]
        
        # Try multiple codecs in order of preference
        codecs = [
            ('avc1', '.mp4'),   # H.264 codec (macOS compatible)
            ('mp4v', '.mp4'),   # MP4V codec
            ('XVID', '.avi'),   # XVID codec (widely supported)
            ('MJPG', '.avi')    # Motion JPEG
        ]
        
        for codec, ext in codecs:
            try:
                # Update output path with correct extension
                current_output = output_path if output_path.endswith(ext) else output_path.rsplit('.', 1)[0] + ext
                
                print(f"Attempting to save video with codec {codec} to {current_output}")
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(
                    current_output,
                    fourcc,
                    fps,
                    (width, height),
                    isColor=True
                )
                
                if not video_writer.isOpened():
                    print(f"Failed to initialize video writer with codec {codec}")
                    continue
                
                # Process and write each frame
                for i, frame in enumerate(frames):
                    # Ensure frame is in correct format
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    
                    # Ensure frame is in BGR color space for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Check if frame needs conversion from RGB to BGR
                        # This is a bit tricky - we can't always know the color space
                        # For safety, we'll always convert to BGR since OpenCV expects it
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    # Ensure frame has the correct dimensions
                    if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
                        print(f"Resizing frame {i} from {frame_bgr.shape[:2]} to {height, width}")
                        frame_bgr = cv2.resize(frame_bgr, (width, height))
                    
                    # Write the frame
                    video_writer.write(frame_bgr)
                
                # Release resources
                video_writer.release()
                
                # Verify the video was created successfully
                if os.path.exists(current_output) and os.path.getsize(current_output) > 0:
                    print(f"Successfully saved video to {current_output}")
                    return
                else:
                    print(f"Video file was created but may be empty: {current_output}")
            
            except Exception as e:
                print(f"Error with codec {codec}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Clean up any failed files
                try:
                    if os.path.exists(current_output):
                        os.remove(current_output)
                except:
                    pass
        
        # If we get here, all codecs failed
        raise RuntimeError(
            "Failed to save video with any supported codec. "
            "Try manually combining the frames using ffmpeg: "
            "ffmpeg -framerate {fps} -i frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4"
        )

    def save_video_ffmpeg(self, frames, output_path, fps=30):
        """
        Save video frames using ffmpeg for better compatibility.
        This requires ffmpeg to be installed on your system.
        
        Args:
            frames: List of numpy arrays (frames)
            output_path: Path to save the video
            fps: Frames per second
        """
        import os
        import cv2
        import subprocess
        import tempfile
        import shutil
        import numpy as np
        
        # Create temporary directory to store frames
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Saving frames to temporary directory: {temp_dir}")
            print(f"Saving frame files: {os.listdir(temp_dir)}")
            
            # Save each frame as an image file
            for i, frame in enumerate(frames):
                # Ensure frame is in correct format (BGR for OpenCV)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                
                # Convert from RGB to BGR if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # Save the frame
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame_bgr)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Build ffmpeg command
            input_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-profile:v", "high",
                "-crf", "23",  # Quality (0-51, lower is better)
                "-pix_fmt", "yuv420p",  # Standard pixel format for compatibility
                output_path
            ]
            
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            try:
                subprocess.run(cmd, check=True)
                print(f"Successfully saved video to {output_path}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error running ffmpeg: {str(e)}")
                # Try to give helpful error information
                if "not found" in str(e) or e.returncode == 127:
                    print("ERROR: ffmpeg command not found. Please install ffmpeg on your system.")
                    print("You can install it with: brew install ffmpeg (on macOS) or apt-get install ffmpeg (on Ubuntu)")
                return False
            
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Video Generator")
    parser.add_argument("--source", required=True, help="Path to source image(s), comma-separated for multiple sources")
    parser.add_argument("--driving", required=True, help="Path to driving video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--background", help="Path to background image (optional)")
    parser.add_argument("--style", help="Path to style reference image (optional)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--expression_intensity", type=float, default=1.0, help="Expression transfer intensity (0.0 to 1.0)")
    parser.add_argument("--no_stabilization", action="store_true", help="Disable video stabilization")
    parser.add_argument("--no_face_swap", action="store_true", help="Disable face swapping")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    args = parser.parse_args()
    
    # Create configuration
    config = AdvancedVideoConfig()
    
    # Create model
    model = AdvancedVideoGenerator(config)
    
    # Load source images
    try:
        # Load and validate source image
        source_paths = args.source.split(',')
        source_images = []
        for source_path in source_paths:
            print(f"Loading source image: {source_path}")
            source_img = model._prepare_image(source_path.strip())
            source_images.append(source_img)
            
            # Validate source image has a face
            source_np = source_img.squeeze(0).permute(1, 2, 0).numpy()
            source_np = (source_np * 255).astype(np.uint8)
            faces = model.face_detector.detect_landmarks(source_np)
            
            # Check if faces were detected using numpy array check
            if faces is None or np.all(faces == 0):  # Check if array is all zeros
                raise ValueError(f"No face detected in source image: {source_path}")
            print(f"Found {len(faces)} face(s) in source image")
            
            # Save debug image
            debug_source = source_np.copy()
            # Only draw landmarks if they exist and aren't all zero
            if faces is not None and not np.all(faces == 0):
                for landmark in faces:
                    if not np.all(landmark == 0):  # Only draw non-zero landmarks
                        x, y = int(landmark[0]), int(landmark[1])
                        cv2.circle(debug_source, (x, y), 1, (0, 255, 0), -1)
            
            debug_path = f"debug_source_{os.path.basename(source_path)}"
            cv2.imwrite(debug_path, cv2.cvtColor(debug_source, cv2.COLOR_RGB2BGR))
            print(f"Saved debug image to {debug_path}")
        
        # Load and validate driving video
        print(f"Loading driving video: {args.driving}")
        driving_frames = model._prepare_video(args.driving, args.max_frames)
        
        if not driving_frames:
            raise ValueError("No valid frames were extracted from the driving video")
        
    except Exception as e:
        print(f"Error during video generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
        
    # Load background image if provided
    background_image = None
    if args.background:
        background_image = model._prepare_image(args.background)
    
    # Load style image if provided
    style_image = None
    if args.style:
        style_image = model._prepare_image(args.style)
    
    # Generate video
    try:
        generated_frames = model.generate_video(
            source_images=source_images,
            driving_frames=driving_frames,
            background_image=background_image,
            style_image=style_image,
            expression_intensity=args.expression_intensity,
            enable_stabilization=not args.no_stabilization,
            swap_faces=not args.no_face_swap
        )
        
        if not generated_frames:
            raise ValueError("No frames were generated")
            
        # Try to save video using both methods
        print("Attempting to save video with ffmpeg...")
        if model.save_video_ffmpeg(generated_frames, args.output, args.fps):
            print("Video saved successfully with ffmpeg!")
        else:
            print("Falling back to OpenCV video writer...")
            model._save_video(generated_frames, args.output, args.fps)
            
        print("Video generation complete.")
        
    except Exception as e:
        print(f"Error during video generation: {str(e)}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()