import os
from hifigan_config import HiFiGANConfig

def verify_hifigan_models():
    """Verify all HiFi-GAN models are present and accessible"""
    print("Verifying HiFi-GAN models...")
    
    for model_name, info in HiFiGANConfig.MODEL_PATHS.items():
        print(f"\nChecking {model_name}:")
        paths = HiFiGANConfig.get_model_path(model_name)
        
        for path_type, path in paths.items():
            if path:  # Skip empty paths (like do_path for non-universal models)
                exists = os.path.exists(path)
                status = "✓ Found" if exists else "✗ Missing"
                print(f"  {path_type}: {status}")
                if not exists:
                    print(f"    Expected at: {path}")

if __name__ == "__main__":
    verify_hifigan_models()