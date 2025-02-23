from core.video_generator import VideoConfig, VideoProcessor

# Initialize
config = VideoConfig(
    image_size=(256, 256),
    num_kp=15,
    block_expansion=64
)
processor = VideoProcessor(config)

# Generate video
processor.generate_video(
    source_image_path='path/to/source/image.jpg',
    driving_video_path='path/to/driving/video.mp4',
    output_path='path/to/output/video.mp4'
)