# 📦 Configuration Settings using Pydantic

from pydantic_settings import BaseSettings
from pathlib import Path
import sys

from app.logger.log import logging  
from app.exception.exception_handler import AppException  

class Settings(BaseSettings):
    """
    ✅ Environment Settings

    Loads environment variables using Pydantic's BaseSettings.
    Supports default values and fetches from a `.env` file.

    Attributes:
        groq_api_key (str): Your Groq API key for accessing Groq services.
        embedding_model (str): Embedding model used for vector generation.
        chroma_persist_directory (str): Directory for ChromaDB persistence.
    """
    groq_api_key: str
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_directory: str = "./chroma_db"

    class Config:
        """
        🔧 Configuration for loading .env

        Path: Two levels up from this file's location.
        """
        env_file = Path(__file__).resolve().parents[2] / ".env"


# 🔄 Load settings with logging and exception handling
try:
    logging.info("🔐 Loading application settings from .env")
    settings = Settings()
    logging.info("✅ Settings loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load settings.", exc_info=True)
    raise AppException(e) from e
