import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class FOMMConfig:
    """Configuration for First Order Motion Model implementation"""
    num_kp: int = 15  # Number of keypoints
    image_size: int = 256  # Input image size
    num_channels: int = 3  # Number of input channels
    block_expansion: int = 64  # Base number of channels for convolutional blocks
    max_features: int = 1024  # Maximum number of features
    num_down_blocks: int = 5  # Number of downsampling blocks
    num_bottleneck_blocks: int = 2  # Number of bottleneck blocks
    use_occlusion: bool = True  # Whether to use occlusion awareness
    use_deformed_source: bool = True  # Whether to use deformed source image
    kp_variance: float = 0.01  # Keypoint Gaussian variance
    enable_stabilization: bool = True  # Enable stabilization for output video

class Encoder(nn.Module):
    """Encoder module to extract features from source and driving images"""
    def __init__(self, config: FOMMConfig):
        super().__init__()
        
        # Initial convolution
        self.conv = nn.Conv2d(config.num_channels, config.block_expansion, kernel_size=7, padding=3)
        
        # Downsampling blocks
        down_blocks = []
        in_features = config.block_expansion
        for i in range(config.num_down_blocks):
            out_features = min(in_features * 2, config.max_features)
            down_blocks.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ))
            in_features = out_features
        self.down_blocks = nn.ModuleList(down_blocks)
        
        # Bottleneck blocks
        bottleneck = []
        for _ in range(config.num_bottleneck_blocks):
            bottleneck.append(nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU()
            ))
        self.bottleneck = nn.Sequential(*bottleneck)
        
        self.out_features = in_features
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out_dict = {}
        
        # Initial convolution
        x = self.conv(x)
        out_dict["initial"] = x
        
        # Downsampling
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            out_dict[f"down_{i+1}"] = x
            
        # Bottleneck
        x = self.bottleneck(x)
        out_dict["bottleneck"] = x
        
        return out_dict

class KeypointDetector(nn.Module):
    """Keypoint detector module for First Order Motion Model"""
    def __init__(self, config: FOMMConfig):
        super().__init__()
        
        self.config = config
        self.encoder = Encoder(config)
        
        # Convert bottleneck features to keypoints
        self.kp_predictor = nn.Conv2d(
            self.encoder.out_features, config.num_kp, kernel_size=3, padding=1
        )
        
        # Predict keypoint variance (confidence)
        self.kp_variance_predictor = nn.Conv2d(
            self.encoder.out_features, config.num_kp, kernel_size=3, padding=1
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Extract features
        encoder_output = self.encoder(x)
        bottleneck = encoder_output["bottleneck"]
        
        # Predict keypoint heatmaps and variances
        heatmap = self.kp_predictor(bottleneck)
        variance = self.kp_variance_predictor(bottleneck)
        variance = F.softplus(variance) + self.config.kp_variance
        
        # Get keypoint coordinates from heatmap
        final_shape = heatmap.shape
        heatmap = heatmap.view(batch_size, self.config.num_kp, -1)
        
        # Convert to probability distribution and sample keypoint positions
        heatmap = F.softmax(heatmap / variance.view(batch_size, self.config.num_kp, 1), dim=2)
        heatmap = heatmap.view(*final_shape)
        
        # Calculate keypoint coordinates
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, heatmap.shape[2]),
            torch.linspace(-1, 1, heatmap.shape[3]),
            indexing='ij'
        )
        grid = torch.stack([grid_w, grid_h], dim=-1).to(x.device)
        
        # Expected value calculation
        kp = torch.zeros((batch_size, self.config.num_kp, 2)).to(x.device)
        for i in range(self.config.num_kp):
            kp[:, i, 0] = torch.sum(grid[:, :, 0] * heatmap[:, i], dim=[1, 2])
            kp[:, i, 1] = torch.sum(grid[:, :, 1] * heatmap[:, i], dim=[1, 2])
            
        return {"keypoints": kp, "heatmap": heatmap, "variance": variance}

class DenseMotionModule(nn.Module):
    """Generate dense motion field from source image and driving keypoints"""
    def __init__(self, config: FOMMConfig):
        super().__init__()
        
        self.config = config
        
        # Feature dimension
        feature_dim = config.num_kp + 1  # +1 for background
        
        # Hourglass network for motion estimation
        hourglass = []
        # Encoding
        hourglass.append(nn.Conv2d(feature_dim * 2, config.block_expansion, kernel_size=3, padding=1))
        hourglass.append(nn.BatchNorm2d(config.block_expansion))
        hourglass.append(nn.ReLU())
        
        in_features = config.block_expansion
        for i in range(config.num_down_blocks):
            out_features = min(in_features * 2, config.max_features)
            hourglass.append(nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1))
            hourglass.append(nn.BatchNorm2d(out_features))
            hourglass.append(nn.ReLU())
            in_features = out_features
            
        # Bottleneck
        for _ in range(config.num_bottleneck_blocks):
            hourglass.append(nn.Conv2d(in_features, in_features, kernel_size=3, padding=1))
            hourglass.append(nn.BatchNorm2d(in_features))
            hourglass.append(nn.ReLU())
            
        # Decoding
        for i in range(config.num_down_blocks):
            out_features = max(in_features // 2, config.block_expansion)
            hourglass.append(nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1))
            hourglass.append(nn.BatchNorm2d(out_features))
            hourglass.append(nn.ReLU())
            in_features = out_features
            
        self.hourglass = nn.Sequential(*hourglass)
        
        # Final convolutional layers
        self.flow = nn.Conv2d(in_features, 2, kernel_size=3, padding=1)
        if config.use_occlusion:
            self.occlusion = nn.Conv2d(in_features, 1, kernel_size=3, padding=1)
            
        if config.use_deformed_source:
            self.deformed_source = nn.Conv2d(in_features, 1, kernel_size=3, padding=1)
            
    def create_heatmap_representation(
        self, kp_driving: torch.Tensor, kp_source: torch.Tensor, image_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Create heatmap representation for keypoints
        Args:
            kp_driving: Keypoints from driving image [B, K, 2]
            kp_source: Keypoints from source image [B, K, 2]
            image_shape: Output spatial dimensions (H, W)
        Returns:
            Heatmap representation [B, K+1, H, W]
        """
        batch_size, num_kp = kp_driving.shape[:2]
        height, width = image_shape
        
        # Create spatial grid
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        grid = torch.stack([grid_w, grid_h], dim=-1).to(kp_driving.device)
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        
        # Create gaussian heatmaps for driving and source keypoints
        driving_heatmap = torch.zeros((batch_size, num_kp, height, width)).to(kp_driving.device)
        source_heatmap = torch.zeros((batch_size, num_kp, height, width)).to(kp_source.device)
        
        for i in range(num_kp):
            # Expand keypoints to match grid dimensions
            kp_driving_i = kp_driving[:, i].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2]
            kp_source_i = kp_source[:, i].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2]
            
            # Calculate gaussian heatmap
            dist_driving = torch.sum((grid - kp_driving_i) ** 2, dim=-1)
            dist_source = torch.sum((grid - kp_source_i) ** 2, dim=-1)
            
            driving_heatmap[:, i] = torch.exp(-dist_driving / (2 * self.config.kp_variance))
            source_heatmap[:, i] = torch.exp(-dist_source / (2 * self.config.kp_variance))
            
        # Add background channel (inverse of keypoint channels)
        driving_bg = 1 - torch.sum(driving_heatmap, dim=1, keepdim=True).clamp(0, 1)
        source_bg = 1 - torch.sum(source_heatmap, dim=1, keepdim=True).clamp(0, 1)
        
        # Concatenate keypoints and background
        driving_heatmap = torch.cat([driving_heatmap, driving_bg], dim=1)
        source_heatmap = torch.cat([source_heatmap, source_bg], dim=1)
        
        return torch.cat([driving_heatmap, source_heatmap], dim=1)
        
    def forward(
        self, source_image: torch.Tensor, kp_driving: torch.Tensor, kp_source: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate dense motion field
        Args:
            source_image: Source image tensor [B, C, H, W]
            kp_driving: Keypoints from driving image [B, K, 2]
            kp_source: Keypoints from source image [B, K, 2]
        Returns:
            Dictionary with flow field, occlusion mask, and deformed source
        """
        batch_size, _, height, width = source_image.shape
        
        # Create heatmap representation
        heatmap = self.create_heatmap_representation(
            kp_driving, kp_source, (height, width)
        )
        
        # Pass through hourglass network
        features = self.hourglass(heatmap)
        
        # Predict optical flow field
        flow = self.flow(features)
        
        # Store the flow for later use (e.g., quality metrics)
        self.last_flow = flow
        
        out_dict = {"flow": flow}
        
        # Predict occlusion mask if enabled
        if hasattr(self, 'occlusion'):
            occlusion = torch.sigmoid(self.occlusion(features))
            out_dict["occlusion"] = occlusion
            
        # Predict deformed source features if enabled
        if hasattr(self, 'deformed_source'):
            deformed_source = torch.sigmoid(self.deformed_source(features))
            out_dict["deformed_source"] = deformed_source
            
        return out_dict

class FOMM(nn.Module):
    """First Order Motion Model for image animation"""
    def __init__(self, config: FOMMConfig):
        super().__init__()
        
        self.config = config
        
        # Components
        self.keypoint_detector = KeypointDetector(config)
        self.dense_motion_network = DenseMotionModule(config)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(
        self, source_image: torch.Tensor, driving_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Animate source image using driving image
        Args:
            source_image: Source image tensor [B, C, H, W]
            driving_image: Driving image tensor [B, C, H, W]
        Returns:
            Dictionary with animated frames and other outputs
        """
        # Extract keypoints from source and driving images
        source_kp = self.keypoint_detector(source_image)
        driving_kp = self.keypoint_detector(driving_image)
        
        # Generate dense motion field
        dense_motion = self.dense_motion_network(
            source_image, driving_kp["keypoints"], source_kp["keypoints"]
        )
        
        # Create sampling grid from flow field
        batch_size, _, height, width = source_image.shape
        flow = dense_motion["flow"]
        
        # Create base sampling grid (normalized coordinates)
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        grid = torch.stack([grid_w, grid_h], dim=-1).to(self.device)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 2]
        
        # Add flow field to base grid
        flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        grid_with_flow = grid + flow_permuted
        
        # Sample from source image using the grid
        warped_image = F.grid_sample(
            source_image, grid_with_flow, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        output_dict = {
            "animated_frame": warped_image,
            "flow": flow,
            "source_keypoints": source_kp["keypoints"],
            "driving_keypoints": driving_kp["keypoints"]
        }
        
        # Add occlusion mask to output if available
        if "occlusion" in dense_motion:
            output_dict["occlusion"] = dense_motion["occlusion"]
            # Apply occlusion mask
            if self.config.use_occlusion:
                output_dict["animated_frame"] = output_dict["animated_frame"] * (1 - dense_motion["occlusion"])
                
        # Add deformed source features if available
        if "deformed_source" in dense_motion:
            output_dict["deformed_source"] = dense_motion["deformed_source"]
                
        return output_dict
        
    def animate(
        self, source_image: torch.Tensor, driving_frames: List[torch.Tensor],
        relative: bool = True, stabilize: bool = True
    ) -> List[torch.Tensor]:
        """
        Animate source image using a sequence of driving frames
        Args:
            source_image: Source image tensor [1, C, H, W]
            driving_frames: List of driving frame tensors [1, C, H, W]
            relative: Whether to use relative keypoint motion
            stabilize: Whether to stabilize the output video
        Returns:
            List of animated frames
        """
        with torch.no_grad():
            # Extract keypoints from source image
            source_kp = self.keypoint_detector(source_image)["keypoints"]
            
            # Initialize list for storing animated frames
            animated_frames = []
            
            # Get reference keypoints for relative mode
            if relative:
                reference_frame = driving_frames[0]
                reference_kp = self.keypoint_detector(reference_frame)["keypoints"]
            
            # Process each driving frame
            for frame in driving_frames:
                driving_kp = self.keypoint_detector(frame)["keypoints"]
                
                # Apply relative transformation if enabled
                if relative:
                    # Calculate transformation from reference keypoints to current driving keypoints
                    driving_kp_relative = self._keypoint_transformation(
                        source_kp, driving_kp, reference_kp, stabilize
                    )
                else:
                    driving_kp_relative = driving_kp
                
                # Generate dense motion field
                dense_motion = self.dense_motion_network(
                    source_image, driving_kp_relative, source_kp
                )
                
                # Create sampling grid from flow field
                batch_size, _, height, width = source_image.shape
                flow = dense_motion["flow"]
                
                # Create base sampling grid (normalized coordinates)
                grid_h, grid_w = torch.meshgrid(
                    torch.linspace(-1, 1, height),
                    torch.linspace(-1, 1, width),
                    indexing='ij'
                )
                grid = torch.stack([grid_w, grid_h], dim=-1).to(self.device)
                grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 2]
                
                # Add flow field to base grid
                flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
                grid_with_flow = grid + flow_permuted
                
                # Sample from source image using the grid
                warped_image = F.grid_sample(
                    source_image, grid_with_flow, 
                    mode='bilinear', padding_mode='border', align_corners=True
                )
                
                # Apply occlusion mask if available
                if "occlusion" in dense_motion and self.config.use_occlusion:
                    warped_image = warped_image * (1 - dense_motion["occlusion"])
                
                animated_frames.append(warped_image)
                
            return animated_frames
    
    def _keypoint_transformation(
        self, source_kp: torch.Tensor, driving_kp: torch.Tensor, 
        reference_kp: torch.Tensor, stabilize: bool = True
    ) -> torch.Tensor:
        """
        Apply relative keypoint transformation
        Args:
            source_kp: Source keypoints [B, K, 2]
            driving_kp: Current driving keypoints [B, K, 2]
            reference_kp: Reference driving keypoints [B, K, 2]
            stabilize: Whether to stabilize head position
        Returns:
            Transformed keypoints [B, K, 2]
        """
        # Calculate keypoint displacements
        driving_to_ref = driving_kp - reference_kp
        
        if stabilize:
            # Stabilize head position by removing mean translation
            driving_to_ref_mean = torch.mean(driving_to_ref, dim=1, keepdim=True)
            driving_to_ref = driving_to_ref - driving_to_ref_mean
        
        # Apply displacement to source keypoints
        transformed_kp = source_kp + driving_to_ref
        
        return transformed_kp