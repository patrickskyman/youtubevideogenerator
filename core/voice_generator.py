import requests
import time
from typing import Optional, Dict
import io
import wave
import json

class VoiceGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
    async def generate_voice(
        self, 
        text: str, 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Default voice ID
        model_id: str = "eleven_monolingual_v1"
    ) -> bytes:
        """Generate voice audio from text using ElevenLabs API."""
        
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
            return response.content
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Voice generation failed: {str(e)}")
