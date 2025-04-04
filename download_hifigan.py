import os
import requests
import hashlib
import json
import sys
import subprocess
from tqdm import tqdm
import re

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, destination, description=None):
    """Download a file with progress bar"""
    try:
        print(f"Downloading {description or destination}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(destination, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, 
            desc=description or os.path.basename(destination)
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def install_gdown():
    """Install gdown package if not already installed"""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown package for Google Drive downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            return True
        except Exception as e:
            print(f"Failed to install gdown: {e}")
            return False

def download_from_gdrive_folder(folder_id, destination_dir, file_pattern=None):
    """Download files from a Google Drive folder"""
    try:
        import gdown
        
        print(f"Attempting to download from Google Drive folder: {folder_id}")
        os.makedirs(destination_dir, exist_ok=True)
        
        # First try folder download
        try:
            output = gdown.download_folder(
                id=folder_id,
                output=destination_dir,
                quiet=False,
                use_cookies=False
            )
            if output:
                print(f"Successfully downloaded folder contents to {destination_dir}")
                return True
        except Exception as e:
            print(f"Folder download failed: {e}")
            
        # If folder download fails, try individual files with direct IDs
        hifigan_file_ids = {
            "generator_universal.pth.tar": "1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y",
            "generator_v1.pth": "1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF",
            "universal_large.pth": "1oNikB8Fz3ABuXQJLNHSKwrfwBoxing9b",
            "g_00400000.pth": "1RHr-pqe8VD8S8Jf29LtiQVAIYKtJYo2v"
        }
        
        downloaded_any = False
        for filename, file_id in hifigan_file_ids.items():
            output_path = os.path.join(destination_dir, filename)
            print(f"Attempting to download {filename} with ID {file_id}")
            try:
                success = gdown.download(
                    id=file_id,
                    output=output_path,
                    quiet=False,
                    use_cookies=False
                )
                if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Successfully downloaded {filename}")
                    downloaded_any = True
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        
        return downloaded_any
                
    except Exception as e:
        print(f"Error with Google Drive folder download: {str(e)}")
        return False

def download_config_file(models_dir):
    """Download HiFi-GAN config file from GitHub"""
    config_urls = [
        "https://raw.githubusercontent.com/jik876/hifi-gan/main/config_v1.json",
        "https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json"
    ]
    
    config_path = os.path.join(models_dir, "config.json")
    
    for url in config_urls:
        if download_file(url, config_path, "HiFi-GAN config file"):
            print(f"Successfully downloaded config file from {url}")
            return True
    
    # If all URLs fail, try to create a default config
    try:
        default_config = {
            "resblock": "1",
            "num_gpus": 1,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
            "seed": 1234,
            "upsample_rates": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "segment_size": 8192,
            "num_mels": 80,
            "num_freq": 1025,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000,
            "fmax_for_loss": None,
            "num_workers": 4,
            "dist_config": {
                "dist_backend": "nccl",
                "dist_url": "tcp://localhost:54321",
                "world_size": 1
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        print("Created default config file as fallback")
        return True
        
    except Exception as e:
        print(f"Failed to create default config: {e}")
        return False

def find_model_files(directory):
    """Find HiFi-GAN model files in directory"""
    model_extensions = ['.pth', '.pt', '.pth.tar']
    models = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                models.append(os.path.join(root, file))
    
    return models

def verify_model_file(file_path):
    """Verify if file is a valid PyTorch model"""
    try:
        import torch
        checkpoint = torch.load(file_path, map_location='cpu')
        return True
    except Exception as e:
        print(f"Invalid model file {file_path}: {e}")
        return False

def download_hifigan():
    """Download and install HiFi-GAN vocoder"""
    # Create models directory
    models_dir = os.path.join('models', 'hifigan')
    create_directory(models_dir)
    
    # Download config file
    config_success = download_config_file(models_dir)
    
    # Install gdown
    if not install_gdown():
        print("Failed to install gdown. Cannot download from Google Drive.")
        return False
    
    # Try to download model files from Google Drive folder
    models_success = download_from_gdrive_folder(
        "1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y", 
        models_dir
    )
    
    if not models_success:
        print("\nAttempting alternative download methods...")
        # Try direct URLs for universal model
        universal_model_urls = [
            "https://github.com/jik876/hifi-gan/releases/download/v1/generator_universal.pth.tar",
            "https://huggingface.co/spaces/junyanz/HiFi-GAN/resolve/main/checkpoints/generator_universal.pth.tar",
            "https://github.com/yluo42/TAL/releases/download/v0.1/hifigan_universal.pt",
        ]
        
        for url in universal_model_urls:
            model_filename = url.split('/')[-1]
            model_path = os.path.join(models_dir, model_filename)
            if download_file(url, model_path, "HiFi-GAN universal model"):
                models_success = True
                break
    
    # Check if we have any model files
    model_files = find_model_files(models_dir)
    if not model_files:
        print("Failed to download any HiFi-GAN model files.")
        return False
    
    # Verify model files
    valid_models = [f for f in model_files if verify_model_file(f)]
    if not valid_models:
        print("No valid HiFi-GAN model files were found.")
        return False
    
    # Create metadata file with paths to models
    metadata = {
        'config_path': os.path.join(models_dir, 'config.json'),
        'model_files': valid_models,
        'primary_model': valid_models[0]  # Use first valid model as primary
    }
    
    with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nHiFi-GAN setup completed!")
    print(f"Models are located in: {os.path.abspath(models_dir)}")
    print(f"Found {len(valid_models)} valid model files:")
    for model in valid_models:
        print(f"  - {os.path.basename(model)}")
    
    return True

def main():
    print("HiFi-GAN Downloader - Enhanced Version")
    print("=====================================")
    
    # Download HiFi-GAN
    success = download_hifigan()
    
    if success:
        print("\nDownload process completed. The models are ready to use.")
        print("You can now run your voice cloner with the HiFi-GAN vocoder.")
    else:
        print("\nDownload process failed. Please check the errors above.")
        print("You may need to manually download the HiFi-GAN models.")

if __name__ == "__main__":
    main()