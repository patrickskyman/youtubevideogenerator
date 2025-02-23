import asyncio
from dotenv import load_dotenv
import os

from utils.content_generator_test import ContentGenerator

async def test_generation():
    load_dotenv()
    
    content_gen = ContentGenerator(
        openai_key=os.getenv("OPENAI_API_KEY"),
        elevenlabs_key=os.getenv("ELEVENLABS_API_KEY")
    )
    
    content = await content_gen.generate_content(
        topic="Why Python is great for beginners",
        duration=30
    )
    
    print(content["script"])

asyncio.run(test_generation())