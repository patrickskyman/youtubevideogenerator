import os
import cv2
import torch
import numpy as np
import warnings
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import face_alignment

# Suppress warnings
warnings.filterwarnings("ignore")

class TalkingHeadAnimator:
    """
    A class for creating talking head animations from a single image
    using First Order Motion Model (FOMM) approach.
    """
    
    def __init__(self, checkpoint_path, config_path):
        """
        Initialize the animator with model paths
        
        Args:
            checkpoint_path: Path to the pretrained model checkpoint
            config_path: Path to the model configuration file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration file
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Initialize face detector
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                              device='cuda' if torch.cuda.is_available() else 'cpu',
                                              flip_input=False)
    
    def _load_model(self, checkpoint_path):
        """Load the FOMM model"""
        from modules.generator import OcclusionAwareGenerator
        from modules.keypoint_detector import KPDetector
        from modules.dense_motion import DenseMotionNetwork
        
        # Initialize Generator
        generator = OcclusionAwareGenerator(**self.config['model_params']['generator_params'],
                                          **self.config['model_params']['common_params'])
        
        # Initialize Keypoint Detector
        kp_detector = KPDetector(**self.config['model_params']['kp_detector_params'],
                               **self.config['model_params']['common_params'])
        
        # Initialize Dense Motion Network
        dense_motion_network = DenseMotionNetwork(**self.config['model_params']['dense_motion_params'],
                                             **self.config['model_params']['common_params'])
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
        
        generator.to(self.device)
        kp_detector.to(self.device)
        dense_motion_network.to(self.device)
        
        generator.eval()
        kp_detector.eval()
        dense_motion_network.eval()
        
        return {
            'generator': generator,
            'kp_detector': kp_detector,
            'dense_motion_network': dense_motion_network
        }
    
    def _preprocess_image(self, img_path):
        """Preprocess the source image"""
        img = imageio.imread(img_path)
        
        # Detect face and get bounding box
        bboxes = self.fa.face_detector.detect_from_image(img.copy())
        if len(bboxes) == 0:
            raise ValueError("No face detected in the source image")
        
        # Use the first detected face
        x1, y1, x2, y2, _ = map(int, bboxes[0])
        
        # Add padding
        h, w = y2 - y1, x2 - x1
        pad = max(h, w) // 4
        y1, y2 = max(0, y1 - pad), min(img.shape[0], y2 + pad)
        x1, x2 = max(0, x1 - pad), min(img.shape[1], x2 + pad)
        
        # Crop face
        face = img[y1:y2, x1:x2]
        
        # Resize to 256x256
        face = resize(face, (256, 256))[..., :3]
        
        # Convert to tensor
        face = torch.tensor(face[np.newaxis].transpose(0, 3, 1, 2)).float().to(self.device)
        
        return face
    
    def _preprocess_driving_video(self, video_path):
        """Preprocess the driving video"""
        driving_frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read frames
        for _ in tqdm(range(frame_count), desc="Processing driving video"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            bboxes = self.fa.face_detector.detect_from_image(frame.copy())
            if len(bboxes) == 0:
                continue
                
            # Use the first detected face
            x1, y1, x2, y2, _ = map(int, bboxes[0])
            
            # Add padding
            h, w = y2 - y1, x2 - x1
            pad = max(h, w) // 4
            y1, y2 = max(0, y1 - pad), min(frame.shape[0], y2 + pad)
            x1, x2 = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
            
            # Crop face
            face = frame[y1:y2, x1:x2]
            
            # Resize to 256x256
            face = resize(face, (256, 256))[..., :3]
            
            driving_frames.append(face)
        
        cap.release()
        
        # Convert to tensor
        driving = torch.tensor(np.array(driving_frames).transpose(0, 3, 1, 2)).float().to(self.device)
        
        return driving, fps
    
    def _generate_keypoints(self, driving, source):
        """Generate keypoints from driving and source images"""
        with torch.no_grad():
            kp_source = self.model['kp_detector'](source)
            kp_driving_initial = self.model['kp_detector'](driving[:1])
            
            kp_driving_list = []
            for frame_idx in tqdm(range(driving.shape[0]), desc="Generating keypoints"):
                driving_frame = driving[frame_idx:frame_idx+1]
                kp_driving = self.model['kp_detector'](driving_frame)
                kp_driving_list.append(kp_driving)
                
        return kp_source, kp_driving_initial, kp_driving_list
    
    def _animate_frame(self, source, kp_source, kp_driving, kp_driving_initial):
        """Animate a single frame"""
        with torch.no_grad():
            # Calculate transform
            kp_norm = self._normalize_kp(kp_source=kp_source,
                                    kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            
            # Generate dense motion
            dense_motion = self.model['dense_motion_network'](source_image=source,
                                                          kp_driving=kp_norm,
                                                          kp_source=kp_source)
            
            # Generate output
            out = self.model['generator'](source_image=source, 
                                       dense_motion=dense_motion)
            
            return out['prediction'].data.cpu().numpy()
    
    def _normalize_kp(self, kp_source, kp_driving, kp_driving_initial):
        """Normalize keypoints"""
        # Get keypoints
        kp_source_value = kp_source['value']
        kp_driving_value = kp_driving['value']
        kp_driving_initial_value = kp_driving_initial['value']
        
        # Calculate transform
        source_mean = torch.mean(kp_source_value, dim=1, keepdim=True)
        driving_mean = torch.mean(kp_driving_value, dim=1, keepdim=True)
        driving_initial_mean = torch.mean(kp_driving_initial_value, dim=1, keepdim=True)
        
        # Apply transform
        kp_norm = kp_driving.copy()
        kp_norm['value'] = kp_driving_value - driving_mean + driving_initial_mean
        
        # Apply relative scaling
        source_std = torch.std(kp_source_value, dim=1, keepdim=True)
        driving_std = torch.std(kp_driving_value, dim=1, keepdim=True)
        driving_initial_std = torch.std(kp_driving_initial_value, dim=1, keepdim=True)
        
        kp_norm['value'] = kp_norm['value'] * source_std / (driving_std + 1e-6) * 0.9
        
        return kp_norm
    
    def animate(self, source_path, driving_path, output_path):
        """
        Animate a source image based on a driving video
        
        Args:
            source_path: Path to the source image
            driving_path: Path to the driving video
            output_path: Path to save the output video
        """
        # Preprocess source image
        source = self._preprocess_image(source_path)
        
        # Preprocess driving video
        driving, fps = self._preprocess_driving_video(driving_path)
        
        # Generate keypoints
        kp_source, kp_driving_initial, kp_driving_list = self._generate_keypoints(driving, source)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))
        
        # Generate animated frames
        for i in tqdm(range(len(kp_driving_list)), desc="Generating animation"):
            kp_driving = kp_driving_list[i]
            
            # Animate frame
            prediction = self._animate_frame(source, kp_source, kp_driving, kp_driving_initial)
            
            # Convert to uint8 and BGR for OpenCV
            frame = (np.transpose(prediction[0], [1, 2, 0]) * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            video_writer.write(frame)
        
        # Release resources
        video_writer.release()
        print(f"Animation saved to {output_path}")
        
        return output_path
    
    def animate_from_audio(self, source_path, audio_path, output_path, temp_driving_path=None):
        """
        Animate a source image based on an audio file
        
        Args:
            source_path: Path to the source image
            audio_path: Path to the audio file
            output_path: Path to save the output video
            temp_driving_path: Path to save the temporary driving video
        """
        # Ensure temporary driving video path exists
        if temp_driving_path is None:
            temp_driving_path = "temp_driving_video.mp4"
        
        # Generate temporary driving video from audio using Wav2Lip
        self._generate_driving_video_from_audio(audio_path, temp_driving_path)
        
        # Animate source image using the generated driving video
        return self.animate(source_path, temp_driving_path, output_path)
    
    def _generate_driving_video_from_audio(self, audio_path, output_path):
        """
        Generate a driving video from an audio file using Wav2Lip
        
        Note: This requires Wav2Lip to be installed and accessible.
        You can modify this method to use a different audio-to-lip-sync approach.
        """
        try:
            from wav2lip import inference as wav2lip_inference
            
            # Define Wav2Lip parameters
            args = {
                "checkpoint_path": "wav2lip_gan.pth",  # Path to Wav2Lip model
                "face": "temp_face.mp4",  # Path to a temporary face video
                "audio": audio_path,
                "outfile": output_path,
                "no_smooth": False,
                "img_size": 96,
                "fps": 25
            }
            
            # Generate driving video
            wav2lip_inference.main(args)
            
        except ImportError:
            print("Wav2Lip not found. Please install it or implement a different audio-to-lip-sync approach.")
            raise
        
        return output_path


# Define a more accessible CLI interface
def main():
    parser = ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--driving", required=True, help="Path to driving video or audio file")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--mode", default="video", choices=["video", "audio"], help="Animation mode")
    parser.add_argument("--checkpoint", default="vox-cpk.pth", help="Path to model checkpoint")
    parser.add_argument("--config", default="config/vox-256.yaml", help="Path to model config")
    
    args = parser.parse_args()
    
    # Initialize animator
    animator = TalkingHeadAnimator(args.checkpoint, args.config)
    
    # Animate
    if args.mode == "video":
        animator.animate(args.source, args.driving, args.output)
    else:  # audio mode
        animator.animate_from_audio(args.source, args.driving, args.output)


# Additional helper functions for image/video processing
class ImageProcessor:
    """Helper class for image preprocessing"""
    
    @staticmethod
    def enhance_portrait(image_path):
        """Enhance portrait image quality"""
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance
        
        # Load image
        img = Image.open(image_path)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        
        # Enhance color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)
        
        # Save enhanced image
        enhanced_path = f"{os.path.splitext(image_path)[0]}_enhanced{os.path.splitext(image_path)[1]}"
        img.save(enhanced_path)
        
        return enhanced_path

    @staticmethod
    def remove_background(image_path):
        """Remove background from portrait image"""
        try:
            from rembg import remove
            from PIL import Image
            
            # Load image
            img = Image.open(image_path)
            
            # Remove background
            img_no_bg = remove(img)
            
            # Save image without background
            no_bg_path = f"{os.path.splitext(image_path)[0]}_no_bg{os.path.splitext(image_path)[1]}"
            img_no_bg.save(no_bg_path)
            
            return no_bg_path
        except ImportError:
            print("rembg not found. Please install it with 'pip install rembg'")
            return image_path


class PretrainedModelManager:
    """Helper class for managing pretrained models"""
    
    @staticmethod
    def download_fomm_model(model_path="vox-cpk.pth"):
        """Download FOMM model"""
        import os
        import requests
        import gdown
        from tqdm import tqdm
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}")
            return model_path
        
        # Try multiple sources
        print(f"Downloading FOMM model to {model_path}...")
        
        # Try direct HuggingFace URL first (more reliable)
        try:
            url = "https://huggingface.co/datasets/trevtravtrev/first-order-model/resolve/main/vox-cpk.pth"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(model_path, 'wb') as f, tqdm(
                    desc=model_path,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            print(f"Successfully downloaded FOMM model to {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            
            # Try alternative Google Drive source with proper ID
            try:
                print("Trying alternative Google Drive source...")
                file_id = "15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
                gdown.download(id=file_id, output=model_path, quiet=False)
                return model_path
            except Exception as e:
                print(f"Failed to download from Google Drive: {e}")
                
                # Try another alternative source
                try:
                    print("Trying second alternative source...")
                    url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/lEw8uRm140L_ag"
                    response = requests.get(url)
                    response.raise_for_status()
                    download_url = response.json()['href']
                    
                    # Download the file
                    response = requests.get(download_url, stream=True)
                    response.raise_for_status()
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    return model_path
                except Exception as e:
                    print(f"Failed to download from all sources: {e}")
                    print("Please download the model manually from https://github.com/AliaksandrSiarohin/first-order-model")
                    return None
    
    @staticmethod
    def download_config(config_path="config/vox-256.yaml"):
        """Download FOMM configuration file"""
        import os
        import requests
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Check if config already exists
        if os.path.exists(config_path):
            print(f"Config already exists at {config_path}")
            return config_path
        
        # Download config
        print(f"Downloading config to {config_path}...")
        url = "https://raw.githubusercontent.com/AliaksandrSiarohin/first-order-model/master/config/vox-256.yaml"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(config_path, 'w') as f:
                f.write(response.text)
            
            print(f"Successfully downloaded config to {config_path}")
            return config_path
            
        except Exception as e:
            print(f"Failed to download config: {e}")
            print("Please download the config manually from https://github.com/AliaksandrSiarohin/first-order-model/blob/master/config/vox-256.yaml")
            
            # Create a basic config as fallback
            try:
                print("Creating basic config as fallback...")
                config_content = """
                dataset_params:
                  root_dir: data/vox-cpu
                  frame_shape: [256, 256, 3]
                  id_sampling: true
                  pairs_list: null
                  augmentation_params:
                    flip_param:
                      horizontal_flip: true
                      time_flip: true
                    jitter_param:
                      brightness: 0.1
                      contrast: 0.1
                      saturation: 0.1
                      hue: 0.1
                
                model_params:
                  common_params:
                    num_kp: 10
                    num_channels: 3
                    estimate_jacobian: true
                  kp_detector_params:
                     temperature: 0.1
                     block_expansion: 32
                     max_features: 1024
                     scale_factor: 0.25
                     num_blocks: 5
                  generator_params:
                    block_expansion: 64
                    max_features: 512
                    num_down_blocks: 2
                    num_bottleneck_blocks: 6
                    estimate_occlusion_map: true
                    dense_motion_params:
                      block_expansion: 64
                      max_features: 1024
                      num_blocks: 5
                      scale_factor: 0.25
                """
                
                with open(config_path, 'w') as f:
                    f.write(config_content)
                
                print(f"Created basic config at {config_path}")
                return config_path
                
            except Exception as e:
                print(f"Failed to create basic config: {e}")
                return None
    
    @staticmethod
    def download_wav2lip_model(model_path="wav2lip_gan.pth"):
        """Download Wav2Lip model"""
        import os
        import requests
        import gdown
        from tqdm import tqdm
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}")
            return model_path
        
        # Try multiple sources
        print(f"Downloading Wav2Lip model to {model_path}...")
        
        # Try HuggingFace URL first (more reliable)
        try:
            url = "https://huggingface.co/spaces/skytnt/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(model_path, 'wb') as f, tqdm(
                    desc=model_path,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            print(f"Successfully downloaded Wav2Lip model to {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            
            # Try alternative Google Drive source
            try:
                print("Trying alternative Google Drive source...")
                file_id = "15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
                gdown.download(id=file_id, output=model_path, quiet=False)
                return model_path
            except Exception as e:
                print(f"Failed to download from all sources: {e}")
                print("Please download the model manually from https://github.com/Rudrabha/Wav2Lip")
                return None

# Example usage
if __name__ == "__main__":
    main()