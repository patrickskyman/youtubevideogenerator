import os
import argparse
from base_new_face_animation_system_model_based import FacialAnimationSystem

def main():
    parser = argparse.ArgumentParser(description='Facial Animation with Wav2Lip')
    parser.add_argument('--video_path', required=True, help='Path to input video file')
    parser.add_argument('--audio_path', required=True, help='Path to audio file for lip syncing')
    parser.add_argument('--output_path', required=True, help='Path for output video')
    parser.add_argument('--wav2lip_model', required=True, help='Path to Wav2Lip checkpoint (.pth file)')
    parser.add_argument('--face_detection_model', default='face_detection/detection/sfd/', 
                        help='Path to face detection model')
    parser.add_argument('--use_original', action='store_true', 
                        help='Use original animation system instead of Wav2Lip')
    
    args = parser.parse_args()
    
    # Initialize animation system
    animation_system = FacialAnimationSystem()
    
    if not args.use_original:
        # Add Wav2Lip support
        success = animation_system.add_wav2lip_support(
            wav2lip_checkpoint_path=args.wav2lip_model,
            face_detection_model_path=args.face_detection_model
        )
        
        if success:
            # Animate with Wav2Lip
            animation_system.animate_video_with_wav2lip(
                video_path=args.video_path,
                audio_path=args.audio_path,
                output_path=args.output_path
            )
        else:
            print("Falling back to original animation system...")
            animation_system.animate_video(
                video_path=args.video_path,
                audio_path=args.audio_path,
                output_path=args.output_path
            )
    else:
        # Use original animation system
        animation_system.animate_video(
            video_path=args.video_path,
            audio_path=args.audio_path,
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()