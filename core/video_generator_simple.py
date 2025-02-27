import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import tensorflow as tf
from PIL import Image
from absl import logging 
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import torch.nn.functional as F
from torchvision import transforms, models
import dlib
from scipy.spatial import Delaunay
import kornia

@dataclass
class AdvancedVideoConfig:
    """Configuration for AdvancedVideoGenerator"""
    # Image settings
    image_size: Tuple[int, int] = (256, 256)
    num_channels: int = 3
    num_kp: int = 478  # MediaPipe Face Mesh landmarks
    
    # Network architecture
    block_expansion: int = 64
    max_features: int = 512
    num_down_blocks: int = 3
    num_bottleneck_blocks: int = 3
    
    # Quality settings
    quality_threshold: float = 0.85
    max_faces: int = 3
    interpolation_factor: int = 2  # Number of frames to interpolate
    
    # Processing settings
    real_time_buffer_size: int = 30
    processing_threads: int = 4

class KeypointDetector(nn.Module):
    """Detect keypoints in the image"""
    def __init__(self, config: AdvancedVideoConfig):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(config.num_channels, config.block_expansion, 
                     kernel_size=7, padding=3),
            nn.BatchNorm2d(config.block_expansion),
            nn.ReLU(),
            *self._make_down_blocks(config),
            *self._make_bottleneck_blocks(config)
        )
        
        self.keypoint_layer = nn.Conv2d(
            config.block_expansion * (2 ** config.num_down_blocks),
            config.num_kp, kernel_size=1
        )
        
    def _make_down_blocks(self, config: AdvancedVideoConfig) -> List[nn.Module]:
        blocks = []
        in_features = config.block_expansion
        for i in range(config.num_down_blocks):
            out_features = min(in_features * 2, config.max_features)
            blocks.extend([
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ])
            in_features = out_features
        return blocks
    
    def _make_bottleneck_blocks(self, config: AdvancedVideoConfig) -> List[nn.Module]:
        blocks = []
        features = min(config.block_expansion * 
                      (2 ** config.num_down_blocks), config.max_features)
        
        for _ in range(config.num_bottleneck_blocks):
            blocks.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU()
            ])
        return blocks
    
    def forward(self, x):
        features = self.encoder(x)
        keypoints = self.keypoint_layer(features)
        return keypoints

class DenseMotionNetwork(nn.Module):
    """Generate dense motion field from source image and driving keypoints"""
    def __init__(self, config: AdvancedVideoConfig):
        super().__init__()
        
        self.hourglass = nn.Sequential(
            # Encoding path
            nn.Conv2d(config.num_channels * 2 + config.num_kp * 2, 
                     config.block_expansion, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.block_expansion),
            nn.ReLU(),
            *self._make_down_blocks(config),
            
            # Bottleneck
            *self._make_bottleneck_blocks(config),
            
            # Decoding path
            *self._make_up_blocks(config)
        )
        
        self.flow = nn.Conv2d(config.block_expansion, 2, kernel_size=3, padding=1)
        
    def _make_down_blocks(self, config: AdvancedVideoConfig) -> List[nn.Module]:
        blocks = []
        in_features = config.block_expansion
        for i in range(config.num_down_blocks):
            out_features = min(in_features * 2, config.max_features)
            blocks.extend([
                nn.Conv2d(in_features, out_features, kernel_size=3, 
                         stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ])
            in_features = out_features
        return blocks
    
    def _make_up_blocks(self, config: AdvancedVideoConfig) -> List[nn.Module]:
        blocks = []
        in_features = min(config.block_expansion * 
                         (2 ** config.num_down_blocks), config.max_features)
        
        for i in range(config.num_down_blocks):
            out_features = max(in_features // 2, config.block_expansion)
            blocks.extend([
                nn.ConvTranspose2d(in_features, out_features, 
                                 kernel_size=3, stride=2, padding=1,
                                 output_padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ])
            in_features = out_features
        return blocks
    
    def _make_bottleneck_blocks(self, config: AdvancedVideoConfig) -> List[nn.Module]:
        blocks = []
        features = min(config.block_expansion * 
                      (2 ** config.num_down_blocks), config.max_features)
        
        for _ in range(config.num_bottleneck_blocks):
            blocks.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU()
            ])
        return blocks
    
    def forward(self, source_image, driving_kp, source_kp):
        # Create heatmap representations of keypoints
        source_heatmap = self._create_heatmap(source_kp, source_image.shape)
        driving_heatmap = self._create_heatmap(driving_kp, source_image.shape)
        
        # Concatenate inputs
        x = torch.cat([source_image, driving_heatmap, source_heatmap], dim=1)
        
        # Generate flow field
        x = self.hourglass(x)
        flow = self.flow(x)
        return flow
    
    def _create_heatmap(self, keypoints, shape):
        # Implementation of keypoint to heatmap conversion
        heatmap = torch.zeros(shape[0], self.num_kp, shape[2], shape[3])
        for i in range(keypoints.shape[1]):
            heatmap[:, i] = self._gaussian_2d(
                shape[2], shape[3], 
                keypoints[:, i, 0], keypoints[:, i, 1]
            )
        return heatmap
    
    def _gaussian_2d(self, height, width, mean_x, mean_y, std=0.1):
        """Create 2D Gaussian heatmap"""
        x = torch.arange(0, width).view(1, -1).repeat(height, 1)
        y = torch.arange(0, height).view(-1, 1).repeat(1, width)
        
        x = x.unsqueeze(0).repeat(mean_x.shape[0], 1, 1)
        y = y.unsqueeze(0).repeat(mean_y.shape[0], 1, 1)
        
        gx = torch.exp(-(x - mean_x.view(-1, 1, 1)) ** 2 / (2 * std ** 2))
        gy = torch.exp(-(y - mean_y.view(-1, 1, 1)) ** 2 / (2 * std ** 2))
        
        return gx * gy
    
class FaceMeshDetector:
    def __init__(self, image_height=256, image_width=256):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.image_width = image_width
        self.image_height = image_height

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        # Ensure image is RGB and within bounds
        if image.shape[0] != self.image_height or image.shape[1] != self.image_width:
            image = cv2.resize(image, (self.image_width, self.image_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        landmarks = np.zeros((478, 3), dtype=np.float32)  # Use 478 3D landmarks (x, y, z)
        
        if results.multi_face_landmarks:
            valid_landmarks = []
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                if idx < 478 and not (np.isnan(landmark.x) or np.isnan(landmark.y) or np.isnan(landmark.z)):
                    valid_landmarks.append([landmark.x, landmark.y, landmark.z])
            if not valid_landmarks:
                print("No valid landmarks detected.")
                return np.zeros((478, 3), dtype=np.float32)
            landmarks = np.array(valid_landmarks, dtype=np.float32)
            if len(landmarks) < 478:
                landmarks = np.pad(landmarks, ((0, 478 - len(landmarks)), (0, 0)), mode='constant')
            return landmarks
        return np.zeros((478, 3), dtype=np.float32)

class QualityAssessor(nn.Module):
    """Assess the quality of generated frames"""
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet for feature extraction
        self.feature_extractor = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        
        # Quality assessment layers
        self.quality_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, generated_frame: torch.Tensor, reference_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated_frame: Shape (C, H, W) or (B, C, H, W)
            reference_frame: Shape (C, H, W) or (B, C, H, W)
        """
        # Add batch dimension if needed
        if len(generated_frame.shape) == 3:
            generated_frame = generated_frame.unsqueeze(0)
        if len(reference_frame.shape) == 3:
            reference_frame = reference_frame.unsqueeze(0)
            
        # Convert to numpy and correct format (B, H, W, C)
        gen_np = generated_frame.permute(0, 2, 3, 1).cpu().numpy()
        ref_np = reference_frame.permute(0, 2, 3, 1).cpu().numpy()
        
        # Ensure values are in [0, 255] range
        if gen_np.max() <= 1.0:
            gen_np = (gen_np * 255).clip(0, 255).astype(np.uint8)
        if ref_np.max() <= 1.0:
            ref_np = (ref_np * 255).astype(np.uint8)
        
        # Extract features using TensorFlow
        gen_features = self.feature_extractor(gen_np)
        ref_features = self.feature_extractor(ref_np)
        
        # Convert TensorFlow tensors to NumPy arrays
        gen_features = gen_features.numpy()
        ref_features = ref_features.numpy()
        
        # Convert to PyTorch tensors
        gen_features = torch.from_numpy(gen_features).float()
        ref_features = torch.from_numpy(ref_features).float()
        
        # Compare features
        feature_diff = torch.abs(gen_features - ref_features)
        
        # Assess quality
        quality_score = self.quality_layers(feature_diff)
        return quality_score

class BackgroundProcessor:
    """Handle background separation and manipulation"""
    def __init__(self):
        # Initialize segmentation model using MobileNetV2
        self.segment_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Add segmentation head
        self.segment_head = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
        ])
        
    def separate_background(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate foreground and background"""
        # Resize image for model
        orig_size = image.shape[:2]
        resized = cv2.resize(image, (224, 224))
        normalized = resized / 255.0
        
        # Generate segmentation mask
        features = self.segment_model(normalized[np.newaxis, ...])
        mask = self.segment_head(features)[0]
        
        # Resize mask back to original size
        mask = cv2.resize(mask.numpy(), (orig_size[1], orig_size[0]))
        mask = (mask > 0.5).astype(np.uint8)
        
        # Refine mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Separate foreground and background
        foreground = image * mask[..., np.newaxis]
        background = image * (1 - mask[..., np.newaxis])
        
        return foreground, background, mask
    
    def replace_background(self, 
                         foreground: np.ndarray, 
                         mask: np.ndarray,
                         new_background: np.ndarray) -> np.ndarray:
        """Replace background with new image"""
        # Resize new background to match foreground
        new_background = cv2.resize(new_background, 
                                  (foreground.shape[1], foreground.shape[0]))
        
        # Blend images
        result = foreground + new_background * (1 - mask[..., np.newaxis])
        return result
    
    def blur_background(self, 
                       foreground: np.ndarray, 
                       background: np.ndarray,
                       mask: np.ndarray,
                       blur_amount: int = 15) -> np.ndarray:
        """Blur the background while keeping foreground sharp"""
        # Apply Gaussian blur to background
        blurred_bg = cv2.GaussianBlur(background, 
                                     (blur_amount, blur_amount), 0)
        
        # Combine with foreground
        result = foreground + blurred_bg * (1 - mask[..., np.newaxis])
        return result


class QualityMetrics:
    """Enhanced quality assessment with multiple metrics"""
    def __init__(self):
        self.landmark_history = []
        self.motion_history = []
        
    def calculate_landmark_stability(self, 
                                   landmarks: np.ndarray, 
                                   window_size: int = 5) -> float:
        """Calculate stability of facial landmarks over time"""
        self.landmark_history.append(landmarks)
        if len(self.landmark_history) > window_size:
            self.landmark_history.pop(0)
            
        if len(self.landmark_history) < 2:
            return 1.0
            
        stability_scores = []
        for i in range(1, len(self.landmark_history)):
            prev = self.landmark_history[i-1]
            curr = self.landmark_history[i]
            movement = np.mean(np.abs(curr - prev))
            stability = 1.0 / (1.0 + movement)
            stability_scores.append(stability)
            
        return np.mean(stability_scores)
    
    def calculate_motion_smoothness(self, 
                                  flow_field: Optional[torch.Tensor],
                                  window_size: int = 5) -> float:
        """Calculate smoothness of motion between frames"""
        # Return default smoothness if no flow field
        if flow_field is None:
            return 1.0
            
        # Convert flow field to numpy
        flow_np = flow_field.cpu().numpy() if isinstance(flow_field, torch.Tensor) else flow_field
        
        self.motion_history.append(flow_np)
        if len(self.motion_history) > window_size:
            self.motion_history.pop(0)
            
        if len(self.motion_history) < 2:
            return 1.0
            
        smoothness_scores = []
        for i in range(1, len(self.motion_history)):
            prev = self.motion_history[i-1]
            curr = self.motion_history[i]
            acceleration = np.mean(np.abs(curr - prev))
            smoothness = 1.0 / (1.0 + acceleration)
            smoothness_scores.append(smoothness)
            
        return np.mean(smoothness_scores)

class FrameInterpolator:
    """Interpolate frames for smoother motion"""
    def __init__(self, config: AdvancedVideoConfig):
        self.config = config
        
    def interpolate(self, 
                   frame1: torch.Tensor, 
                   frame2: torch.Tensor) -> List[torch.Tensor]:
        """Generate intermediate frames"""
        interpolated = []
        for i in range(self.config.interpolation_factor):
            t = (i + 1) / (self.config.interpolation_factor + 1)
            # Linear interpolation of features
            inter_frame = frame1 * (1 - t) + frame2 * t
            interpolated.append(inter_frame)
        return interpolated

class MultiFaceProcessor:
    """Handle multiple faces in video"""
    def __init__(self, config: AdvancedVideoConfig):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.max_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect and extract multiple faces"""
        results = self.face_mesh.process(image)
        faces = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.zeros((468, 3))
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks[idx] = [landmark.x, landmark.y, landmark.z]
                faces.append(landmarks)
                
        return faces
    
    def process_multiple_faces(self, 
                             source_images: List[np.ndarray],
                             driving_frame: np.ndarray) -> List[np.ndarray]:
        """Process multiple faces in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.processing_threads) as executor:
            futures = []
            for source_image in source_images:
                future = executor.submit(
                    self._process_single_face,
                    source_image,
                    driving_frame
                )
                futures.append(future)
            
            results = [future.result() for future in futures]
        return results
    
    def _process_single_face(self, 
                           source_image: np.ndarray,
                           driving_frame: np.ndarray) -> np.ndarray:
        """Process a single face (to be run in parallel)"""
        # Implementation of single face processing
        # This would use the core video generation logic
        pass

class RealTimeProcessor:
    """Handle real-time video processing"""
    def __init__(self, config: AdvancedVideoConfig):
        self.config = config
        self.frame_buffer = queue.Queue(maxsize=config.real_time_buffer_size)
        self.result_buffer = queue.Queue(maxsize=config.real_time_buffer_size)
        self.processing_thread = None
        self.running = False
        
    def start(self, generator: 'AdvancedVideoGenerator'):
        """Start real-time processing"""
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_frames,
            args=(generator,)
        )
        self.processing_thread.start()
        
    def stop(self):
        """Stop real-time processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the processing queue"""
        try:
            self.frame_buffer.put(frame, timeout=1/30)  # 30 fps timeout
        except queue.Full:
            # Skip frame if buffer is full
            pass
        
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get next processed frame"""
        try:
            return self.result_buffer.get_nowait()
        except queue.Empty:
            return None
        
    def _process_frames(self, generator: 'AdvancedVideoGenerator'):
        """Frame processing loop"""
        while self.running:
            try:
                frame = self.frame_buffer.get(timeout=1/30)
                processed = generator.process_single_frame(frame)
                self.result_buffer.put(processed)
            except queue.Empty:
                continue
            
class EnhancedFaceSwapper:
    """Improved face swapping between source and target images with enhanced color correction and special eye handling"""
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        
    def swap_faces(self, source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """Enhanced face swapping with better blending, color correction and triangle warping"""
        print("Starting enhanced face swap...")
        
        # 1-9: [Keep existing code until after creation of result_img]
        
        # 1. Validate and prepare images
        if source_img is None or target_img is None:
            print("Error: One or both input images are None")
            return target_img if target_img is not None else np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Ensure 3-channel images
        if len(source_img.shape) != 3 or source_img.shape[2] != 3:
            source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR) if len(source_img.shape) == 2 else cv2.cvtColor(source_img, cv2.COLOR_BGRA2BGR)
        if len(target_img.shape) != 3 or target_img.shape[2] != 3:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR) if len(target_img.shape) == 2 else cv2.cvtColor(target_img, cv2.COLOR_BGRA2BGR)
            
        # 2. Detect faces with Dlib (more reliable for detection)
        source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        source_faces = self.face_detector(source_gray)
        target_faces = self.face_detector(target_gray)
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            print(f"Found {len(source_faces)} face(s) in source image and {len(target_faces)} face(s) in target image")
            return target_img
            
        print(f"Found {len(source_faces)} face(s) in source image")
        print(f"Found {len(target_faces)} face(s) in target image")
        
        # 3. Get facial landmarks using Dlib (68 points)
        source_shape = self.shape_predictor(source_gray, source_faces[0])
        target_shape = self.shape_predictor(target_gray, target_faces[0])
        
        source_landmarks = np.array([[p.x, p.y] for p in source_shape.parts()], dtype=np.float32)
        target_landmarks = np.array([[p.x, p.y] for p in target_shape.parts()], dtype=np.float32)
        
        # Create debug visualization with better landmark visibility
        debug_img = target_img.copy()
        for point in target_landmarks:
            x, y = int(point[0]), int(point[1])
            # Use larger circles and add outline for better visibility
            cv2.circle(debug_img, (x, y), 3, (0, 0, 0), -1)  # Black outline
            cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)  # Green center
        cv2.imwrite('debug_dlib_landmarks.jpg', debug_img)
        
        # 4. Extract eye landmarks and create eye masks
        left_eye_src = source_landmarks[36:42]
        right_eye_src = source_landmarks[42:48]
        left_eye_tgt = target_landmarks[36:42]
        right_eye_tgt = target_landmarks[42:48]
        
        # Create eye masks
        left_eye_mask_src = np.zeros(source_img.shape[:2], dtype=np.uint8)
        right_eye_mask_src = np.zeros(source_img.shape[:2], dtype=np.uint8)
        left_eye_mask_tgt = np.zeros(target_img.shape[:2], dtype=np.uint8)
        right_eye_mask_tgt = np.zeros(target_img.shape[:2], dtype=np.uint8)
        
        # Draw eye regions with padding
        cv2.fillConvexPoly(left_eye_mask_src, self._expand_eye_region(left_eye_src), 255)
        cv2.fillConvexPoly(right_eye_mask_src, self._expand_eye_region(right_eye_src), 255)
        cv2.fillConvexPoly(left_eye_mask_tgt, self._expand_eye_region(left_eye_tgt), 255)
        cv2.fillConvexPoly(right_eye_mask_tgt, self._expand_eye_region(right_eye_tgt), 255)
        
        # Combine eye masks
        eyes_mask_src = cv2.bitwise_or(left_eye_mask_src, right_eye_mask_src)
        eyes_mask_tgt = cv2.bitwise_or(left_eye_mask_tgt, right_eye_mask_tgt)
        
        # Slightly dilate for better coverage
        eyes_mask_src = cv2.dilate(eyes_mask_src, None, iterations=2)
        eyes_mask_tgt = cv2.dilate(eyes_mask_tgt, None, iterations=2)
        
        # Save eye masks for debugging
        cv2.imwrite('debug_eyes_mask.jpg', eyes_mask_tgt)
        
        # 5. Calculate convex hull for face masking with extended area
        # Get face boundary landmarks (chin and forehead)
        jaw_points = source_landmarks[0:17]  # Jawline landmarks
        forehead_points = self._estimate_forehead(source_landmarks)
        source_boundary = np.vstack([jaw_points, forehead_points])
        
        target_jaw = target_landmarks[0:17]
        target_forehead = self._estimate_forehead(target_landmarks)
        target_boundary = np.vstack([target_jaw, target_forehead])
        
        # Define eye and nose indices for later use
        eye_nose_indices = set()
        for i in range(36, 48):  # Eye landmarks
            eye_nose_indices.add(i)
        for i in range(27, 36):  # Nose landmarks
            eye_nose_indices.add(i)

        # Create the hull using extended points
        source_hull = cv2.convexHull(source_boundary.astype(np.int32))
        target_hull = cv2.convexHull(target_boundary.astype(np.int32))
        
        # 6. Create face masks with feathering for better blending
        source_mask = np.zeros(source_img.shape[:2], dtype=np.uint8)
        target_mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        
        cv2.fillConvexPoly(source_mask, source_hull, 255)
        cv2.fillConvexPoly(target_mask, target_hull, 255)
        
        # Create a gradient mask for feathered blending
        # Create nose mask for special handling
        nose_mask = self._create_nose_mask(target_landmarks, target_mask.shape)

        # Enhanced transition mask with special nose handling
        feather_amount = 15  # Must be odd number
        if feather_amount % 2 == 0:
            feather_amount += 1  # Ensure odd number

        
        # Distance transform for gradient blending
        dist = cv2.distanceTransform(target_mask, cv2.DIST_L2, 5)
        # Normalize to 0-255 range and create gradient
        cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
        
        # Create a feathered mask by thresholding the distance transform
        _, feathered_mask = cv2.threshold(dist, 255 - feather_amount, 255, cv2.THRESH_BINARY)
        feathered_mask = feathered_mask.astype(np.uint8)
        
        # Create nose mask for special handling
        nose_mask = self._create_nose_mask(target_landmarks, target_mask.shape)
        
        # Enhanced transition mask with special nose handling
        transition_mask = cv2.GaussianBlur(
            target_mask, 
            (feather_amount, feather_amount), 
            feather_amount//2
        )
        
        # Strengthen the transition in the nose area
        nose_blur_size = min(31, feather_amount * 2 + 1)  # Ensure odd number and reasonable size
        nose_transition = cv2.GaussianBlur(
            nose_mask, 
            (nose_blur_size, nose_blur_size), 
            nose_blur_size//4
        )

        # Give more weight to the nose region in the transition mask
        transition_mask = cv2.addWeighted(transition_mask, 0.7, nose_transition, 0.3, 0)

        # Save masks for debugging
        cv2.imwrite('enhanced_source_mask.jpg', source_mask)
        cv2.imwrite('enhanced_target_mask.jpg', target_mask)
        cv2.imwrite('enhanced_feathered_mask.jpg', feathered_mask)
        cv2.imwrite('enhanced_transition_mask.jpg', transition_mask)
        cv2.imwrite('enhanced_nose_mask.jpg', nose_mask)
        
        # Continue with triangulation, warping, etc. (steps 7-9 of original code)
        # 7. Delaunay triangulation with better triangle edge handling
        rect = cv2.boundingRect(target_landmarks.astype(np.int32))
        subdiv = cv2.Subdiv2D(rect)
        
        for point in target_landmarks:
            if not np.any(np.isnan(point)) and not np.all(point == 0):
                subdiv.insert(tuple(map(float, point)))
                
        triangles = subdiv.getTriangleList()
        triangle_indices = []
        
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            # Convert triangle points to landmark indices
            idx1 = self._find_closest_landmark(target_landmarks, pt1)
            idx2 = self._find_closest_landmark(target_landmarks, pt2)
            idx3 = self._find_closest_landmark(target_landmarks, pt3)
            
            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangle_indices.append([idx1, idx2, idx3])
        
        print(f"Generated {len(triangle_indices)} triangles using Delaunay triangulation")
        
        # 8. Initialize result image
        result_img = np.copy(target_img)
        
        # 9. Transform triangles from source to target with special handling for eye regions
        for triangle in triangle_indices:
            # Check if this triangle is part of an eye region
            is_eye_triangle = False
            for idx in triangle:
                if 36 <= idx <= 47:  # Eye region landmarks
                    is_eye_triangle = True
                    break
                    
            # Get triangle points
            source_tri = source_landmarks[triangle]
            target_tri = target_landmarks[triangle]
            
            # Apply special handling for eye triangles
            if is_eye_triangle:
                # Skip eye triangles completely to avoid duplication
                continue
            else:
                # Warp triangle normally for non-eye regions
                self._warp_triangle_enhanced(source_img, result_img, source_tri, target_tri)
        
        # 10. Apply advanced color correction before blending with improved nose handling
        color_matched = self._advanced_color_correction(result_img, target_img, transition_mask)
        
        # 11. Create a mask that excludes eyes to avoid double-eye effect
        blend_mask = cv2.bitwise_and(transition_mask, cv2.bitwise_not(eyes_mask_tgt))
        
        # Alpha blend for areas that aren't eyes
        alpha_mask = blend_mask.astype(float) / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)
        
        # Initialize output with alpha blending
        output = color_matched * alpha_mask + target_img * (1 - alpha_mask)
        output = output.astype(np.uint8)
        
        # NEW CODE: Create a specialized mask for nose blending
        nose_points = target_landmarks[27:36]  # Nose landmarks
        nose_blend_mask = np.zeros_like(target_mask)
        nose_hull = cv2.convexHull(nose_points.astype(np.int32))
        cv2.fillConvexPoly(nose_blend_mask, nose_hull, 255)
        
        # Apply more aggressive feathering to nose region
        nose_blend_kernel = 31  # Increased kernel size for smoother transition
        nose_blend_mask = cv2.dilate(nose_blend_mask, None, iterations=3)
        nose_blend_mask = cv2.GaussianBlur(
            nose_blend_mask, 
            (nose_blend_kernel, nose_blend_kernel), 
            nose_blend_kernel//4
        )

        # Apply additional blending to nose region with reduced intensity
        nose_alpha = nose_blend_mask.astype(float) / 255.0 * 0.4  # Reduced from 0.6 to 0.4
        nose_alpha = np.expand_dims(nose_alpha, axis=2)

        # Create a gradual transition zone
        transition_zone = cv2.dilate(nose_blend_mask, None, iterations=5)
        transition_zone = cv2.GaussianBlur(transition_zone, (51, 51), 10)
        transition_alpha = (transition_zone.astype(float) / 255.0 * 0.3)
        transition_alpha = np.expand_dims(transition_alpha, axis=2)

        # Apply two-stage blending
        # First stage: Main nose region
        output = output * (1 - nose_alpha) + target_img * nose_alpha
        # Second stage: Transition zone
        output = output * (1 - transition_alpha) + target_img * transition_alpha
        output = output.astype(np.uint8)
        
        # 12. Apply seamless clone only to central face area (excluding eyes and nose)
        inner_mask = np.zeros_like(target_mask)
        
        # Use central face landmarks for seamless cloning
        # Exclude eye landmarks (36-47) and nose landmarks (27-36)
        central_landmarks = np.vstack([
            target_landmarks[0:27],  # Jaw, eyebrows, etc.
            target_landmarks[48:]    # Mouth, etc.
        ])
        
        central_hull = cv2.convexHull(central_landmarks.astype(np.int32))
        cv2.fillConvexPoly(inner_mask, central_hull, 255)
        
        # Further exclude eye regions from inner mask
        inner_mask = cv2.bitwise_and(inner_mask, cv2.bitwise_not(eyes_mask_tgt))
        # Also exclude nose region from seamless cloning
        inner_mask = cv2.bitwise_and(inner_mask, cv2.bitwise_not(nose_blend_mask))
        inner_mask = cv2.dilate(inner_mask, None, iterations=5)
        inner_mask = cv2.GaussianBlur(inner_mask, (15, 15), 5)
        
        # Calculate center avoiding eyes and nose
        center = np.mean(central_landmarks, axis=0).astype(np.int32)
        
        # Apply seamless clone only to non-eye, non-nose regions
        try:
            if np.sum(inner_mask) > 0:  # Only if mask has content
                output = cv2.seamlessClone(
                    color_matched,
                    output,
                    inner_mask,
                    tuple(center),
                    cv2.NORMAL_CLONE
                )
        except cv2.error:
            print("Seamless cloning failed, using alpha blending only")
        
        # 13. Special handling for eye regions - use the target's original eyes
        for tgt_eye_mask in [left_eye_mask_tgt, right_eye_mask_tgt]:
            # Feather the edges of eye masks for smoother blending
            tgt_eye_mask_feathered = cv2.GaussianBlur(tgt_eye_mask, (5, 5), 2)
            eye_alpha = tgt_eye_mask_feathered.astype(float) / 255.0
            eye_alpha = np.expand_dims(eye_alpha, axis=2)
            
            # Use original target eyes
            output = output * (1 - eye_alpha) + target_img * eye_alpha
        
        # Ensure correct types before final harmonization
        output = output.astype(np.uint8)
        target_img = target_img.astype(np.uint8)
        transition_mask = transition_mask.astype(np.uint8)

        # 14. Final skin tone harmonization with improved nose handling
        output = self._harmonize_skin_tones(output, target_img, transition_mask, target_landmarks)
        
        # Create final debug visualization
        debug_final = output.copy()
        for point in target_landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(debug_final, (x, y), 3, (0, 0, 0), -1)  # Black outline
            cv2.circle(debug_final, (x, y), 2, (0, 255, 0), -1)  # Green center
        cv2.imwrite('debug_enhanced_final.jpg', debug_final)
        
        print("Enhanced face swap completed successfully")
        return output.astype(np.uint8)
    
    def _expand_eye_region(self, eye_points, padding=3):
        """Expand eye region by adding padding to ensure full coverage"""
        # Calculate eye center
        eye_center = np.mean(eye_points, axis=0)
        
        # Expand points outward from center
        expanded_points = []
        for point in eye_points:
            # Vector from center to point
            vector = point - eye_center
            # Normalize vector
            length = np.sqrt(np.sum(vector ** 2))
            if length > 0:
                unit_vector = vector / length
                # Expand point outward
                expanded_point = point + unit_vector * padding
                expanded_points.append(expanded_point)
            else:
                expanded_points.append(point)
                
        return np.array(expanded_points, dtype=np.int32)
        
    def _estimate_forehead(self, landmarks):
        """Estimate forehead points above the eyebrows for better face mask"""
        # Get eyebrow points
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        
        # Calculate forehead points by extrapolating above eyebrows
        forehead_points = []
        for point in np.vstack([left_eyebrow, right_eyebrow]):
            # Move point up to estimate forehead
            forehead_x = point[0]
            forehead_y = point[1] - 20  # Move up by 20 pixels, adjust as needed
            forehead_points.append([forehead_x, forehead_y])
            
        return np.array(forehead_points, dtype=np.float32)
        
    def _find_closest_landmark(self, landmarks, point):
        """Find the closest landmark index to a given point with improved accuracy"""
        distances = np.sqrt(np.sum((landmarks - point) ** 2, axis=1))
        min_dist_idx = np.argmin(distances)
        
        # Use a threshold to ensure accurate matching (adjust as needed)
        if distances[min_dist_idx] < 5:  # Reduced threshold for better precision
            return min_dist_idx
        return None
        
    def _warp_triangle_enhanced(self, src_img, dst_img, src_tri, dst_tri):
        """Enhanced triangle warping with better error handling and blending"""
        # Get bounding rectangle for destination triangle
        rect = cv2.boundingRect(dst_tri.astype(np.int32))
        (x, y, w, h) = rect
        
        # Check if rectangle is within image bounds with better margin handling
        if x < 0 or y < 0 or x + w > dst_img.shape[1] or y + h > dst_img.shape[0]:
            # Adjust rectangle to fit within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, dst_img.shape[1] - x)
            h = min(h, dst_img.shape[0] - y)
            
            # If dimensions are too small, skip this triangle
            if w <= 1 or h <= 1:
                return
        # Offset triangles by the rectangular region
        dst_tri_cropped = np.array([
            [dst_tri[0][0] - x, dst_tri[0][1] - y],
            [dst_tri[1][0] - x, dst_tri[1][1] - y],
            [dst_tri[2][0] - x, dst_tri[2][1] - y]
        ], dtype=np.float32)
        
        src_tri_cropped = np.array([
            [src_tri[0][0], src_tri[0][1]],
            [src_tri[1][0], src_tri[1][1]],
            [src_tri[2][0], src_tri[2][1]]
        ], dtype=np.float32)
        
        # Create mask for destination triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_cropped.astype(np.int32), 255)
        
        # Apply slight blur to mask edges for smoother blending
        mask = cv2.GaussianBlur(mask, (3, 3), 1)
        
        # Warp source triangle to match destination
        try:
            warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
        except cv2.error:
            # Handle degenerate triangle case
            return
            
        # Warp with better interpolation for smoother results
        warped = cv2.warpAffine(
            src_img, 
            warp_mat, 
            (w, h), 
            flags=cv2.INTER_LANCZOS4,  # Better interpolation
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Apply mask to keep only the triangle region
        warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
        
        # Create inverse mask for the original content
        mask_inv = cv2.bitwise_not(mask)
        
        # Extract the region from destination image
        dst_region = dst_img[y:y+h, x:x+w]
        
        tri_area = cv2.contourArea(dst_tri_cropped.astype(np.int32))
        if tri_area < 10:  # Skip very small triangles
            return
        
        if np.mean(mask) < 5:  # If mask has very few white pixels
            return
        
        # Ensure dimensions match before applying mask
        if dst_region.shape[:2] != mask_inv.shape:
            # Resize mask to match destination region
            mask_inv = cv2.resize(mask_inv, (dst_region.shape[1], dst_region.shape[0]))
            
        # Apply mask to destination
        try:
            original_cropped = cv2.bitwise_and(dst_region, dst_region, mask=mask_inv)
            
            # Combine original and warped triangles with alpha blending for smoother transition
            alpha = mask.astype(float) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            # Ensure dimensions match before combining
            if original_cropped.shape == warped_masked.shape and alpha.shape[:2] == warped_masked.shape[:2]:
                blended = warped_masked * alpha + original_cropped * (1 - alpha)
                dst_img[y:y+h, x:x+w] = blended.astype(np.uint8)
            else:
                # Fallback to simpler combination if dimensions don't match
                dst_img[y:y+h, x:x+w] = cv2.add(original_cropped, warped_masked)
        except cv2.error:
            # Handle any remaining errors silently
            pass
            
    def _advanced_color_correction(self, source, target, mask):
        """Advanced color correction with face-aware adjustments"""
        # Create masked versions of source and target
        mask_3ch = cv2.merge([mask, mask, mask])
        source_face = cv2.bitwise_and(source, mask_3ch)
        target_face = cv2.bitwise_and(target, mask_3ch)
        
        # Convert to LAB color space for better color matching
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # Calculate statistics for each channel in the LAB space
        for i in range(3):  # L, A, B channels
            # Get masked arrays for statistics calculation
            source_channel = source_lab[:, :, i]
            target_channel = target_lab[:, :, i]
            
            # Get mean and std of both source and target
            source_mean, source_std = cv2.meanStdDev(source_channel, mask=mask)
            target_mean, target_std = cv2.meanStdDev(target_channel, mask=mask)
            
            # Adjust source channel to match target statistics
            # formula: new = (old - mean_old) * (std_target / std_source) + mean_target
            if source_std[0][0] > 0:
                alpha = target_std[0][0] / source_std[0][0]
            else:
                alpha = 1.0
                
            # Apply correction to entire channel
            source_lab[:, :, i] = np.clip(
                (source_channel - source_mean[0][0]) * alpha + target_mean[0][0],
                0, 255
            ).astype(np.uint8)
        
        # Convert back to BGR
        corrected = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
        
        # Apply only to masked region using alpha blending
        alpha = mask.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        result = corrected * alpha + source * (1 - alpha)
        
        return result.astype(np.uint8)
        
    def _create_nose_mask(self, landmarks, shape):
        """Create a specialized mask for the nose region with better transition"""
        # Get nose and surrounding landmarks
        nose_points = landmarks[27:36]  # Main nose landmarks
        upper_lip_points = landmarks[48:55]  # Upper lip points for better transition
        
        # Create extended nose points to include area below nose
        extended_points = np.vstack([
            nose_points,
            upper_lip_points,
            landmarks[31:36]  # Add bottom nose points again for stronger weight
        ])
        
        # Create initial nose mask
        nose_mask = np.zeros(shape, dtype=np.uint8)
        
        # Create hull for the extended region
        nose_hull = cv2.convexHull(extended_points.astype(np.int32))
        cv2.fillConvexPoly(nose_mask, nose_hull, 255)
        
        # Create a gradual falloff mask
        falloff_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(falloff_mask, nose_hull, 255)
        
        # Apply progressive blurring for smoother transition
        for i in range(3):
            kernel_size = 2 * i + 3  # Increasing kernel sizes: 3, 5, 7
            falloff_mask = cv2.GaussianBlur(falloff_mask, (kernel_size, kernel_size), 0)
        
        # Normalize the falloff mask
        falloff_mask = falloff_mask.astype(float) / 255.0
        
        # Create final mask with smooth transition
        final_mask = (nose_mask.astype(float) * falloff_mask).astype(np.uint8)
        
        return final_mask

    def _harmonize_skin_tones(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray, target_landmarks=None) -> np.ndarray:
        """Final skin tone harmonization for natural appearance with improved nose region handling"""
        # Ensure inputs are uint8
        if source.dtype != np.uint8:
            source = (source * 255).clip(0, 255).astype(np.uint8)
        if target.dtype != np.uint8:
            target = (target * 255).clip(0, 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).clip(0, 255).astype(np.uint8)

        # Convert to YCrCb color space for better skin tone handling
        try:
            source_ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
            target_ycrcb = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e:
            print(f"Color conversion error: {str(e)}")
            print(f"Source shape: {source.shape}, dtype: {source.dtype}")
            print(f"Target shape: {target.shape}, dtype: {target.dtype}")
            return source

        # Create special nose mask if landmarks are provided
        nose_mask = None
        if target_landmarks is not None:
            nose_mask = self._create_nose_mask(target_landmarks, mask.shape)
            # Save nose mask for debugging
            cv2.imwrite('debug_nose_mask.jpg', nose_mask)
        
        # Reduce intensity of harmonization to prevent oily appearance
        harmonization_strength = 0.7  # Adjust this value between 0-1
        
        # For the chrominance channels only (Cr and Cb)
        for i in range(1, 3):
            # Create gradual transition for chrominance channels
            transition_factor = mask.astype(float) / 255.0 * harmonization_strength
            
            # Apply stronger blending factor for nose region if available
            if nose_mask is not None:
                nose_factor = nose_mask.astype(float) / 255.0 * 0.9  # Stronger blending for nose
                # Combine regular transition with nose-specific blending
                transition_factor = np.maximum(transition_factor, nose_factor)
            
            # Apply weighted average for chrominance
            source_ycrcb[:, :, i] = (
                source_ycrcb[:, :, i] * (1 - transition_factor) + 
                target_ycrcb[:, :, i] * transition_factor
            ).astype(np.uint8)
        
        # Also adjust luminance (Y channel) specifically in the nose region for better matching
        if nose_mask is not None:
            nose_lum_factor = nose_mask.astype(float) / 255.0 * 0.6  # Less aggressive for luminance
            source_ycrcb[:, :, 0] = (
                source_ycrcb[:, :, 0] * (1 - nose_lum_factor) + 
                target_ycrcb[:, :, 0] * nose_lum_factor
            ).astype(np.uint8)

        # Convert back to BGR
        try:
            harmonized = cv2.cvtColor(source_ycrcb, cv2.COLOR_YCrCb2BGR)
        except cv2.error as e:
            print(f"Color conversion back error: {str(e)}")
            return source

        # Apply reduced blending at boundaries
        kernel = np.ones((5,5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        boundary_mask = dilated_mask - eroded_mask

        # Create smoother transitions
        boundary_factor = boundary_mask.astype(float) / 255.0 * 0.5
        boundary_factor = np.expand_dims(boundary_factor, axis=2)

        # Apply boundary blending
        result = harmonized.copy()
        result = (result * (1 - boundary_factor) + target * boundary_factor).astype(np.uint8)
        
        # Extra blending pass specifically for nose region if available
        if nose_mask is not None:
            # Create a more focused mask for the problematic seam area
            seam_area = cv2.dilate(nose_mask, kernel, iterations=1) - cv2.erode(nose_mask, kernel, iterations=2)
            seam_factor = seam_area.astype(float) / 255.0 * 0.7  # Stronger for seam area
            seam_factor = np.expand_dims(seam_factor, axis=2)
            
            # Apply targeted blending to problematic area
            result = (result * (1 - seam_factor) + target * seam_factor).astype(np.uint8)

        return result

class StyleTransfer(nn.Module):
    """Neural style transfer for faces"""
    def __init__(self):
        super().__init__()
        # Update VGG19 initialization to use new weights parameter
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]
    
    def transfer_style(self, content_img: torch.Tensor, 
                      style_img: torch.Tensor, 
                      num_steps: int = 300) -> torch.Tensor:
        """Transfer style from style_img to content_img"""
        # Initialize target image
        target = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([target])
        
        # Get features
        content_features = self(content_img)
        style_features = self(style_img)
        
        # Style transfer optimization
        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                target_features = self(target)
                
                content_loss = F.mse_loss(
                    target_features[2], content_features[2]
                )
                
                style_loss = 0
                for tf, sf in zip(target_features, style_features):
                    style_loss += self._gram_loss(tf, sf)
                
                total_loss = content_loss + style_loss
                total_loss.backward()
                return total_loss
                
            optimizer.step(closure)
            
        return target
    
    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def _gram_loss(self, target_feature, style_feature):
        target_gram = self._gram_matrix(target_feature)
        style_gram = self._gram_matrix(style_feature)
        return F.mse_loss(target_gram, style_gram)

class ExpressionTransfer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True
        )

    def extract_expression(self, image: np.ndarray) -> mp.solutions.face_mesh.FaceMesh:
        """Extract facial expression from image"""
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if image is already RGB
            if np.array_equal(image[:,:,0], cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[0]):
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be BGR/RGB with 3 channels")

        # Create MediaPipe Image directly from numpy array
        try:
            # Process image directly without creating mp.Image
            results = self.face_mesh.process(rgb_image)
            return results
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def transfer_expression(self, 
                        source_img: np.ndarray, 
                        target_img: np.ndarray, 
                        intensity: float = 1.0) -> np.ndarray:
        """Transfer expression from source to target with improved implementation"""
        # Convert images to uint8 if needed
        if source_img.dtype != np.uint8:
            source_img = (source_img * 255).clip(0, 255).astype(np.uint8)
        if target_img.dtype != np.uint8:
            target_img = (target_img * 255).clip(0, 255).astype(np.uint8)

        # Extract expressions
        source_expr = self.extract_expression(source_img)
        target_expr = self.extract_expression(target_img)

        if not source_expr.multi_face_landmarks or not target_expr.multi_face_landmarks:
            return target_img
            
        # Get landmarks
        source_landmarks = source_expr.multi_face_landmarks[0].landmark
        target_landmarks = target_expr.multi_face_landmarks[0].landmark
        
        # Create modified landmarks with expression transfer
        modified_landmarks = []
        for i, (s_lm, t_lm) in enumerate(zip(source_landmarks, target_landmarks)):
            # Calculate the difference in expression
            dx = (s_lm.x - t_lm.x) * intensity
            dy = (s_lm.y - t_lm.y) * intensity
            dz = (s_lm.z - t_lm.z) * intensity
            
            # Create new landmark with transferred expression
            new_lm = type(s_lm)()
            new_lm.x = t_lm.x + dx
            new_lm.y = t_lm.y + dy
            new_lm.z = t_lm.z + dz
            modified_landmarks.append(new_lm)
        
        # Convert landmarks to numpy array for warping
        target_points = np.float32([[lm.x * target_img.shape[1], lm.y * target_img.shape[0]] 
                                for lm in target_landmarks])
        modified_points = np.float32([[lm.x * target_img.shape[1], lm.y * target_img.shape[0]] 
                                    for lm in modified_landmarks])
        
        # Create warping transform
        transform = cv2.estimateAffinePartial2D(target_points, modified_points)[0]
        
        # Apply warping
        if transform is not None:
            result_img = cv2.warpAffine(
                target_img,
                transform,
                (target_img.shape[1], target_img.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
            return result_img
            
        return target_img
    
    def _calculate_smile_intensity(self, landmarks) -> float:
        """
        Calculate smile intensity based on mouth corner positions and curvature.
        Returns a value between 0.0 (no smile) and 1.0 (maximum smile).
        """
        # Mouth corner indices (left: 61, right: 291)
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        # Center of top lip (middle point: 0)
        top_lip = landmarks[0]
        
        # Calculate vertical distances of corners relative to center
        left_height = left_corner.y - top_lip.y
        right_height = right_corner.y - top_lip.y
        
        # Calculate mouth width
        mouth_width = abs(right_corner.x - left_corner.x)
        
        # Calculate mouth curvature (average height of corners relative to center)
        curvature = (left_height + right_height) / 2
        
        # Calculate smile ratio (curvature relative to width)
        smile_ratio = curvature / mouth_width if mouth_width > 0 else 0
        
        # Normalize to 0-1 range (empirically determined thresholds)
        normalized_smile = max(0.0, min(1.0, (smile_ratio + 0.15) / 0.3))
        
        return normalized_smile

    def _calculate_eye_openness(self, landmarks) -> float:
        """
        Calculate eye openness based on vertical distance between eyelids.
        Returns a value between 0.0 (closed) and 1.0 (fully open).
        """
        # Left eye indices (top: 159, bottom: 145)
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        
        # Right eye indices (top: 386, bottom: 374)
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        # Calculate vertical distances
        left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
        
        # Calculate eye widths for normalization
        left_eye_width = abs(landmarks[133].x - landmarks[33].x)
        right_eye_width = abs(landmarks[362].x - landmarks[263].x)
        
        # Calculate openness ratios
        left_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        right_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        
        # Average the ratios and normalize to 0-1 range
        avg_ratio = (left_ratio + right_ratio) / 2
        normalized_openness = max(0.0, min(1.0, avg_ratio / 0.35))
        
        return normalized_openness

    def _calculate_mouth_openness(self, landmarks) -> float:
        """
        Calculate mouth openness based on vertical distance between lips.
        Returns a value between 0.0 (closed) and 1.0 (fully open).
        """
        # Upper lip indices (middle: 13)
        upper_lip = landmarks[13]
        
        # Lower lip indices (middle: 14)
        lower_lip = landmarks[14]
        
        # Mouth corners for width reference
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        # Calculate vertical distance between lips
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        # Calculate mouth width for normalization
        mouth_width = abs(right_corner.x - left_corner.x)
        
        # Calculate openness ratio
        openness_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Normalize to 0-1 range
        normalized_openness = max(0.0, min(1.0, openness_ratio / 0.7))
        
        return normalized_openness

    def _calculate_brow_raise(self, landmarks) -> float:
        """
        Calculate eyebrow raise intensity based on distance from eyes.
        Returns a value between 0.0 (neutral) and 1.0 (maximum raise).
        """
        # Left eyebrow keypoints (middle: 52)
        left_brow = landmarks[52]
        left_eye = landmarks[159]  # Upper eyelid
        
        # Right eyebrow keypoints (middle: 282)
        right_brow = landmarks[282]
        right_eye = landmarks[386]  # Upper eyelid
        
        # Calculate vertical distances between brows and eyes
        left_distance = abs(left_brow.y - left_eye.y)
        right_distance = abs(right_brow.y - right_eye.y)
        
        # Calculate eye widths for normalization
        left_eye_width = abs(landmarks[133].x - landmarks[33].x)
        right_eye_width = abs(landmarks[362].x - landmarks[263].x)
        
        # Calculate raise ratios
        left_ratio = left_distance / left_eye_width if left_eye_width > 0 else 0
        right_ratio = right_distance / right_eye_width if right_eye_width > 0 else 0
        
        # Average the ratios and normalize to 0-1 range
        avg_ratio = (left_ratio + right_ratio) / 2
        normalized_raise = max(0.0, min(1.0, (avg_ratio - 0.3) / 0.4))
        
        return normalized_raise

class VideoStabilizer:
    """Stabilize video frames"""
    def __init__(self, smoothing_window: int = 30):
        self.smoothing_window = smoothing_window
        self.prev_gray = None
        self.prev_pts = None
        self.transforms = []
        
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=200, qualityLevel=0.01,
                minDistance=30, blockSize=3
            )
            return frame
            
        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None
        )
        
        # Filter valid points
        if curr_pts is None:  # Add check for None
            return frame
            
        idx = np.where(status == 1)[0]
        if len(idx) < 4:  # Need at least 4 points for transform
            return frame
            
        prev_pts = self.prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Calculate transform
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        if m is None:  # Add check for None
            return frame
            
        # Update transform history
        self.transforms.append(m)
        if len(self.transforms) > self.smoothing_window:
            self.transforms.pop(0)
            
        # Calculate smoothed transform
        if len(self.transforms) > 0:  # Only calculate mean if we have transforms
            smoothed = np.mean(self.transforms, axis=0)
        else:
            return frame
            
        # Apply smoothed transform
        rows, cols = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame, smoothed, (cols, rows)
        )
        
        # Update previous frame info
        self.prev_gray = gray
        self.prev_pts = curr_pts.reshape(-1, 1, 2)
        
        return stabilized

# Update AdvancedVideoGenerator class
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
                results['frames'].extend([
                    frame.cpu().numpy() for frame in interpolated
                ])
            
            # 7. Video Stabilization
            if enable_stabilization:
                processed_frame = self.video_stabilizer.stabilize_frame(
                    processed_frame
                )
            
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
    
# Example usage
def main():
    config = AdvancedVideoConfig(
        max_faces=3,
        interpolation_factor=2,
        real_time_buffer_size=30
    )
    
    generator = AdvancedVideoGenerator(config)
    
    use_webcam = False # Set to True for webcam testing
   
    if use_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            # Load and prepare source image
            source_img = cv2.imread('source.jpg')
            if source_img is None:
                raise FileNotFoundError("Could not load source.jpg")
            
            print("Loading source image...")
            print(f"Source image shape: {source_img.shape}")
            
            # Process source image
            source_img = cv2.resize(source_img, (256, 256))
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            source_img = source_img.astype(np.uint8)  # Ensure uint8 type
            
            print(f"Processed source image shape: {source_img.shape}, dtype: {source_img.dtype}")
            
            # Start real-time processing with source image
            generator.source_image = source_img  # Set source image directly
            
            print("Starting real-time processing. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame = cv2.resize(frame, (256, 256))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)  # Ensure uint8 type
                
                # Process frame
                processed = generator.process_single_frame(frame)
                
                if processed is not None:
                    # Convert back to BGR for display
                    if processed.max() <= 1.0:
                        processed = (processed * 255).astype(np.uint8)
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Processed', processed_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            generator.stop_real_time()
            cap.release()
            cv2.destroyAllWindows()
            
    
    else:
        # Static image processing
        if not use_webcam:
            try:
                # Load images
                source_img = cv2.imread('source.jpg')
                driving_img = cv2.imread('driving.jpg')
                
                if source_img is None or driving_img is None:
                    raise FileNotFoundError("Could not load source.jpg or driving.jpg")
                
                # Validate faces in both images
                face_detector = dlib.get_frontal_face_detector()
                
                # Check source image
                source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
                source_faces = face_detector(source_gray)
                if not source_faces:
                    raise ValueError("No face detected in source.jpg")
                print(f"Found {len(source_faces)} face(s) in source image")
                
                # Check driving image
                driving_gray = cv2.cvtColor(driving_img, cv2.COLOR_BGR2GRAY)
                driving_faces = face_detector(driving_gray)
                if not driving_faces:
                    raise ValueError("No face detected in driving.jpg")
                print(f"Found {len(driving_faces)} face(s) in driving image")
                
                # Process images consistently
                target_size = (256, 256)
                source_img = cv2.resize(source_img, target_size)
                driving_img = cv2.resize(driving_img, target_size)
                
                # Convert to RGB
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                driving_img = cv2.cvtColor(driving_img, cv2.COLOR_BGR2RGB)
                
                # Save debug images
                cv2.imwrite('debug_source.jpg', cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite('debug_driving.jpg', cv2.cvtColor(driving_img, cv2.COLOR_RGB2BGR))
                
                print("Converting to tensors...")
                # Convert to tensors [0-1] range
                source_tensor = torch.from_numpy(source_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                driving_tensor = torch.from_numpy(driving_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                
                print(f"Source tensor shape: {source_tensor.shape}")
                print(f"Driving tensor shape: {driving_tensor.shape}")
                
                print("Starting face swap...")
                results = generator(
                    source_images=source_tensor,
                    driving_frames=[driving_tensor],
                    expression_intensity=0.8,
                    enable_stabilization=True,
                    swap_faces=True
                )
                
                if results['frames']:
                    processed_frame = results['frames'][0]
                    if isinstance(processed_frame, torch.Tensor):
                        processed_frame = processed_frame.cpu().numpy()
                    processed_frame = (processed_frame * 255).clip(0, 255).astype(np.uint8)
                    processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    
                    # Save debug landmarks
                    if results.get('landmarks'):
                        debug_img = processed_bgr.copy()
                        for landmarks in results['landmarks'][0]:
                            for point in landmarks:
                                x, y = int(point[0]), int(point[1])
                                cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
                        cv2.imwrite('debug_landmarks.jpg', debug_img)
                    
                    # Save the result
                    cv2.imwrite('output.jpg', processed_bgr)
                    print("Result saved as output.jpg")
                    
                    # Display the result
                    cv2.imshow('Processed Frame', processed_bgr)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
            except Exception as e:
                print(f"Error processing images: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()