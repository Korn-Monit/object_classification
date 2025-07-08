from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_URI: str = "gs://object-classification/prod/models/v1.0.0/mobilevit_xxs_cifar10.pth"  # Default to prod
    GCP_PROJECT_ID: str = "your-gcp-project"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"