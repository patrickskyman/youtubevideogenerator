import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import argparse
from typing import List, Dict, Tuple, Optional, Union
import torch.nn.functional as F
from core.essentials import (
    AdvancedVideoConfig, 
    BackgroundProcessor, 
    DenseMotionNetwork, 
    ExpressionTransfer, 
    FaceMeshDetector, 
    FrameInterpolator, 
    KeypointDetector, 
    MultiFaceProcessor, 
    QualityAssessor, 
    QualityMetrics, 
    RealTimeProcessor, 
    StyleTransfer,
    VideoStabilizer
)
from core.enhanced_face_swapper import EnhancedFaceSwapper
from core.video_generator import AdvancedVideoGenerator

def prepare_image(image_path: str, config: AdvancedVideoConfig) -> torch.Tensor:
    """
    Load and preprocess an image for the video generation pipeline.
    
    Args:
        image_path: Path to the source image
        config: Configuration for image preprocessing
        
    Returns:
        Preprocessed image as a tensor
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to target dimensions
    image = cv2.resize(image, config.image_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def prepare_video(video_path: str, config: AdvancedVideoConfig, max_frames: int = None) -> List[torch.Tensor]:
    """
    Load and preprocess a video for the generation pipeline.
    
    Args:
        video_path: Path to the driving video
        config: Configuration for video preprocessing
        max_frames: Maximum number of frames to load
        
    Returns:
        List of preprocessed frames as tensors
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, config.image_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        
        frames.append(frame_tensor)
        
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    
    video.release()
    return frames

def generate_video(
    model: AdvancedVideoGenerator,
    source_images: Union[torch.Tensor, List[torch.Tensor]],
    driving_frames: List[torch.Tensor],
    background_image: Optional[torch.Tensor] = None,
    style_image: Optional[torch.Tensor] = None,
    expression_intensity: float = 1.0,
    enable_stabilization: bool = True,
    swap_faces: bool = True
) -> List[np.ndarray]:
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
        results = model(
            source_images=source_images,
            driving_frames=driving_frames,
            background_image=background_image,
            style_image=style_image,
            expression_intensity=expression_intensity,
            enable_stabilization=enable_stabilization,
            swap_faces=swap_faces
        )
    
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

def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Save frames as a video file.
    
    Args:
        frames: List of video frames to save
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
    """
    if not frames:
        raise ValueError("No frames to save")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        video_writer.write(frame)
    
    # Release resources
    video_writer.release()
    print(f"Video saved to {output_path}")

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
    source_paths = args.source.split(',')
    source_images = []
    for source_path in source_paths:
        source_images.append(prepare_image(source_path.strip(), config))
    
    # Load driving video
    driving_frames = prepare_video(args.driving, config, args.max_frames)
    
    # Load background image if provided
    background_image = None
    if args.background:
        background_image = prepare_image(args.background, config)
    
    # Load style image if provided
    style_image = None
    if args.style:
        style_image = prepare_image(args.style, config)
    
    # Generate video
    generated_frames = generate_video(
        model=model,
        source_images=source_images,
        driving_frames=driving_frames,
        background_image=background_image,
        style_image=style_image,
        expression_intensity=args.expression_intensity,
        enable_stabilization=not args.no_stabilization,
        swap_faces=not args.no_face_swap
    )
    
    # Save video
    save_video(generated_frames, args.output, args.fps)
    
    print("Video generation complete.")

if __name__ == "__main__":
    main()