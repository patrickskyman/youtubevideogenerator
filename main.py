from config import settings
from fastapi import FastAPI
from core.video_generator import VideoGenerator
from core.script_generator import ScriptGenerator
from utils.content_generator_test import ContentGenerator
# ... other imports

app = FastAPI()

@app.post("/generate_video/")
async def generate_video(topic: str):
    # Initialize components
    script_gen = ScriptGenerator(settings.OPENAI_API_KEY)
    video_gen = VideoGenerator().to(settings.DEVICE)
    
    # Generate script
    script = await script_gen.generate_script(topic)
    
    # Generate voice
    voice = await generate_voice(script)
    
    # Generate video
    video = video_gen(source_image, driving_sequence)
    
    # Synchronize audio and video
    final_video = sync_audio_with_video(voice, video)
    
    return {"video_url": save_and_get_url(final_video)}

async def main():
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    # Initialize generator
    content_gen = ContentGenerator(
        openai_key=os.getenv("OPENAI_API_KEY"),
        elevenlabs_key=os.getenv("ELEVENLABS_API_KEY")
    )
    
    # Generate content
    try:
        content = await content_gen.generate_content(
            topic="5 Python tips for beginners",
            duration=60
        )
        
        # Print script
        print("Generated Script:")
        print(json.dumps(content["script"].dict(), indent=2))
        
        # Save audio segments
        for i, (segment_type, audio_data) in enumerate(content["audio_segments"]):
            with open(f"audio_segment_{i}_{segment_type}.wav", "wb") as f:
                f.write(audio_data)
                
    except Exception as e:
        print(f"Error generating content: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())