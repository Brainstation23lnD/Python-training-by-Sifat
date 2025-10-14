from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    DATABASE_URL: str

    EMBEDDING_MODEL_NAME: str
    EMBEDDING_DIM: int
    LLM_SERVER_URL: str
    PG_VECTOR_DIM: int
    SECRET_KEY: str
    REDIS_URL: str
    POSTGRES_USER: str
    POSTGRES_DB: str
    POSTGRES_PASSWORD: str

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        extra = "allow"


settings = Settings()
