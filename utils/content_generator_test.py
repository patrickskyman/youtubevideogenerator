# Utility class to combine script and voice generation
from typing import Dict
from core.script_generator import ScriptGenerator
from core.voice_generator import VoiceGenerator


class ContentGenerator:
    def __init__(self, openai_key: str, elevenlabs_key: str):
        self.script_generator = ScriptGenerator(openai_key)
        self.voice_generator = VoiceGenerator(elevenlabs_key)
        
    async def generate_content(self, topic: str, duration: int = 60) -> Dict:
        """Generate both script and voice content."""
        # Generate script
        script = await self.script_generator.generate_script(topic, duration)
        
        # Generate voice for each part separately for better timing control
        audio_segments = []
        
        # Generate hook audio
        hook_audio = await self.voice_generator.generate_voice(script.hook)
        audio_segments.append(("hook", hook_audio))
        
        # Generate main content audio
        for segment in script.main_content:
            audio = await self.voice_generator.generate_voice(segment["text"])
            audio_segments.append(("main", audio))
            
        # Generate CTA audio
        cta_audio = await self.voice_generator.generate_voice(script.call_to_action)
        audio_segments.append(("cta", cta_audio))
        
        return {
            "script": script,
            "audio_segments": audio_segments
        }
