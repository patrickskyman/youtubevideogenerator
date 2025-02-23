from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ELEVENLABS_API_KEY: str
    MODEL_PATH: str
    DEVICE: str = "cuda"
    
    class Config:
        env_file = ".env"