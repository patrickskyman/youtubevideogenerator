from openai import OpenAI
from typing import Dict, List, Optional
import json
import asyncio
from pydantic import BaseModel

class ScriptFormat(BaseModel):
    title: str
    hook: str
    main_content: List[Dict[str, str]]
    call_to_action: str

class ScriptGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    async def generate_script(self, topic: str, duration_seconds: int = 60) -> ScriptFormat:
        """Generate a structured script for a YouTube Short."""
        system_prompt = """
        You are a YouTube Shorts script writer. Create engaging, concise scripts that:
        1. Start with a powerful hook (first 3 seconds)
        2. Deliver value quickly
        3. Include appropriate pauses for timing
        4. End with a clear call-to-action
        
        Format the script as JSON with the following structure:
        {
            "title": "Catchy title",
            "hook": "Attention-grabbing opening line",
            "main_content": [
                {"text": "script segment", "timing": "duration in seconds"}
            ],
            "call_to_action": "Engaging CTA"
        }
        """
        
        user_prompt = f"""
        Create a {duration_seconds}-second YouTube Short script about {topic}.
        Ensure the main_content segments total to {duration_seconds-6} seconds
        (leaving 3 seconds each for hook and CTA).
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            script_data = json.loads(response.choices[0].message.content)
            return ScriptFormat(**script_data)
            
        except Exception as e:
            raise Exception(f"Script generation failed: {str(e)}")