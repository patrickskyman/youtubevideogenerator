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
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)

        # Convert to BGR if in RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=200, 
                qualityLevel=0.01,
                minDistance=30, 
                blockSize=3
            )
            return frame
            
        # Calculate optical flow
        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=200, 
                qualityLevel=0.01,
                minDistance=30, 
                blockSize=3
            )
            return frame

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Filter valid points
        if curr_pts is None:
            return frame
            
        idx = np.where(status == 1)[0]
        if len(idx) < 4:
            return frame
            
        prev_pts = self.prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Calculate transform
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        if m is None:
            return frame
            
        # Update transform history
        self.transforms.append(m)
        if len(self.transforms) > self.smoothing_window:
            self.transforms.pop(0)
            
        # Calculate smoothed transform
        if len(self.transforms) > 0:
            smoothed = np.mean(self.transforms, axis=0)
        else:
            return frame
            
        # Apply smoothed transform
        rows, cols = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame, smoothed, (cols, rows),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Update previous frame info
        self.prev_gray = gray
        self.prev_pts = curr_pts.reshape(-1, 1, 2)
        
        return stabilized

    
