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
    num_kp: int = 468  # MediaPipe Face Mesh landmarks
    
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
    """MediaPipe Face Mesh implementation for precise facial landmark detection"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Extract facial landmarks using MediaPipe Face Mesh"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        landmarks = np.zeros((468, 3))  # 468 3D landmarks
        if results.multi_face_landmarks:
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks[idx] = [landmark.x, landmark.y, landmark.z]
        
        return landmarks

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

class FaceSwapper:
    """Handle face swapping between source and target"""
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
        
    def extract_face_features(self, image: np.ndarray):
        """Extract facial features and landmarks"""
        print("Input image shape:", image.shape)
        print("Input image dtype:", image.dtype)
        
        # Ensure image is uint8 and has 3 channels
        if image.dtype != np.uint8:
            # Fix the dtype error here
            image = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Handle different image formats
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        if len(image.shape) == 4:  # (batch, channels, height, width)
            image = image.squeeze(0).transpose(1, 2, 0)
        elif len(image.shape) == 3 and image.shape[0] == 3:  # (channels, height, width)
            image = image.transpose(1, 2, 0)
            
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] != 3:
            raise ValueError(f"Invalid number of channels: {image.shape[2]}")
            
        # Rest of the function remains the same
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        if not faces:
            return None, None
            
        landmarks = self.shape_predictor(gray, faces[0])
        
        if hasattr(self, 'face_recognition_model'):
            face_descriptor = self.face_recognition_model.compute_face_descriptor(image, landmarks)
            return face_descriptor, self._landmarks_to_np(landmarks)
            
        return None, self._landmarks_to_np(landmarks)
    
    def _landmarks_to_np(self, landmarks):
        """Convert dlib landmarks to numpy array"""
        return np.array([[p.x, p.y] for p in landmarks.parts()])
    
    def _get_triangle_points(self, points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        """Get points forming a triangle"""
        triangle_points = []
        for i in range(3):
            point_idx = triangle[i]
            triangle_points.append(points[point_idx])
        return np.array(triangle_points, dtype=np.float32)

    def ensure_three_channels(self, image):
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            print("Image Shape:", image.shape)
            print("Image Dtype:", image.dtype)
        elif image.shape[2] == 4:  # Image with alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            print("Image Shape:", image.shape)
            print("Image Dtype:", image.dtype)

        return image


    def swap_faces(self, source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """Perform face swapping"""
        source_img = self.ensure_three_channels(source_img)

        source_descriptor, source_landmarks = self.extract_face_features(source_img)
        target_descriptor, target_landmarks = self.extract_face_features(target_img)
        
        if source_landmarks is None or target_landmarks is None:
            return target_img
            
        # Create Delaunay triangulation for face mesh
        rect = cv2.boundingRect(target_landmarks)
        subdiv = cv2.Subdiv2D(rect)
        
        # Add points to subdivision
        points = np.array(target_landmarks, np.int32)
        for point in points:
            subdiv.insert((int(point[0]), int(point[1])))
            
        # Get triangles and convert to numpy array
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        
        # Convert triangle indices to point indices
        triangle_indices = []
        for triangle in triangles:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            
            # Find indices of points in landmarks array
            idx1 = self._find_point_index(points, pt1)
            idx2 = self._find_point_index(points, pt2)
            idx3 = self._find_point_index(points, pt3)
            
            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangle_indices.append([idx1, idx2, idx3])
        
        # Warp triangles from source to target
        result_img = target_img.copy()
        for triangle_idx in triangle_indices:
            source_tri = source_landmarks[triangle_idx]
            target_tri = target_landmarks[triangle_idx]
            self._warp_triangle(source_img, result_img, source_tri, target_tri)
        
        # Create face mask for seamless cloning
        hull = cv2.convexHull(target_landmarks)
        mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Apply seamless cloning
        center = (rect[0] + rect[2]//2, rect[1] + rect[3]//2)
        result_img = cv2.seamlessClone(
            result_img, target_img, mask, center, cv2.NORMAL_CLONE
        )
        
        return result_img
        
    def _find_point_index(self, points: np.ndarray, point: tuple) -> Optional[int]:
        """Find index of point in landmarks array"""
        distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] < 1:  # Threshold for point matching
            return min_dist_idx
        return None
        
    def _warp_triangle(self, 
                      src_img: np.ndarray, 
                      dst_img: np.ndarray, 
                      src_tri: np.ndarray, 
                      dst_tri: np.ndarray) -> None:
        """Warp triangular region from source to destination"""
        # Get bounding box of destination triangle
        rect = cv2.boundingRect(dst_tri)
        (x, y, w, h) = rect
        
        # Offset points by rect topleft
        dst_tri_cropped = dst_tri - [x, y]
        src_tri_cropped = src_tri - [x, y]
        
        # Get mask for triangular region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), 1)
        
        # Apply warp to triangular region
        warp_mat = cv2.getAffineTransform(
            np.float32(src_tri_cropped), 
            np.float32(dst_tri_cropped)
        )
        warped = cv2.warpAffine(
            src_img, 
            warp_mat, 
            (w, h), 
            None, 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        )
        warped = warped * mask[:, :, np.newaxis]
        
        # Copy triangular region to destination image
        dst_img[y:y+h, x:x+w] = dst_img[y:y+h, x:x+w] * (1 - mask[:, :, np.newaxis])
        dst_img[y:y+h, x:x+w] = dst_img[y:y+h, x:x+w] + warped

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
            min_detection_confidence=0.5
        )

    def extract_expression(self, image: np.ndarray) -> np.ndarray:
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

        results = self.face_mesh.process(rgb_image)
        return results

    def transfer_expression(self, 
                          source_img: np.ndarray, 
                          target_img: np.ndarray, 
                          intensity: float = 1.0) -> np.ndarray:
        """Transfer expression from source to target"""
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
            
        # Calculate expression differences
        expr_diff = {k: (source_expr[k] - target_expr[k]) * intensity 
                    for k in source_expr.keys()}
        
        # Apply expression differences to target landmarks
        results = self.face_mesh.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return target_img
            
        landmarks = results.multi_face_landmarks[0].landmark
        modified_landmarks = self._apply_expression_diff(landmarks, expr_diff)
        
        # Warp target image according to modified landmarks
        return self._warp_image(target_img, landmarks, modified_landmarks)
    
    def _calculate_smile_intensity(self, landmarks) -> float:
        # Implementation of smile intensity calculation
        pass
    
    def _calculate_eye_openness(self, landmarks) -> float:
        # Implementation of eye openness calculation
        pass
    
    def _calculate_mouth_openness(self, landmarks) -> float:
        # Implementation of mouth openness calculation
        pass
    
    def _calculate_brow_raise(self, landmarks) -> float:
        # Implementation of brow raise calculation
        pass

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
        self.face_swapper = FaceSwapper()
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
    
    def process_single_frame(self, 
                           frame: np.ndarray,
                           expression_intensity: float = 1.0,
                           enable_stabilization: bool = True) -> np.ndarray:
        """Process a single frame for real-time use"""
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
        
        # Detect faces
        faces = self.multi_face_processor.detect_faces(frame)
        
        # Process each face
        processed_frame = frame.copy()
        for face_idx, face_landmarks in enumerate(faces):
            if hasattr(self, 'source_image') and face_idx < len(self.source_image):
                # Apply face swap
                processed_frame = self.face_swapper.swap_faces(
                    self.source_image[face_idx],
                    processed_frame
                )
                
                # Apply expression transfer
                if expression_intensity > 0:
                    processed_frame = self.expression_transfer.transfer_expression(
                        self.source_image[face_idx],
                        processed_frame,
                        expression_intensity
                    )
        
        # Apply style transfer if enabled
        if self.current_style_image is not None:
            processed_frame = self._apply_style_transfer(processed_frame)
        
        # Apply stabilization if enabled
        if enable_stabilization:
            processed_frame = self.video_stabilizer.stabilize_frame(
                processed_frame
            )
        
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
    
    use_webcam = True  # Set to True for webcam testing
   
    if use_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            # Load and prepare source image
            source_img = cv2.imread('source.jpg')
            if source_img is None:
                raise FileNotFoundError("Could not load source.jpg")
            
            # Process source image
            source_img = cv2.resize(source_img, (256, 256))
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            source_img = source_img.astype(np.uint8)  # Ensure uint8 type
            
            # Start real-time processing with source image
            generator.start_real_time([source_img])  # Pass numpy array directly
            
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
        try:
            source_img = cv2.imread('source.jpg')
            driving_img = cv2.imread('driving.jpg')
            
            if source_img is None or driving_img is None:
                raise FileNotFoundError("Could not load source.jpg or driving.jpg")
            
            # Process images consistently
            target_size = (256, 256)
            source_img = cv2.resize(source_img, target_size)
            driving_img = cv2.resize(driving_img, target_size)
            
            # Convert to RGB
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            driving_img = cv2.cvtColor(driving_img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensors [0-1] range
            source_tensor = torch.from_numpy(source_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            driving_tensor = torch.from_numpy(driving_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            print("Processing images...")
            print(f"Source tensor shape: {source_tensor.shape}")
            print(f"Driving tensor shape: {driving_tensor.shape}")
            
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