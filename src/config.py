from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()