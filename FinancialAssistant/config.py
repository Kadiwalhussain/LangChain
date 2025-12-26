"""Configuration management for Financial Assistant."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str = ""
    
    # LLM Configuration
    llm_provider: str = "ollama"  # openai or ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    openai_model: str = "gpt-4o-mini"
    
    # Embedding Configuration
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Vector Store
    vector_store_path: str = "./database/chroma_db"
    collection_name: str = "financial_docs"
    
    # Directories
    upload_dir: str = "./data/user_uploads"
    knowledge_base_dir: str = "./data/knowledge_base"
    tax_forms_dir: str = "./data/tax_forms"
    
    # Application
    debug: bool = True
    log_level: str = "INFO"
    max_file_size_mb: int = 50
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    temperature: float = 0.2
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create directories if they don't exist
def create_directories():
    """Create necessary directories."""
    settings = Settings()
    directories = [
        settings.upload_dir,
        settings.knowledge_base_dir,
        settings.tax_forms_dir,
        settings.vector_store_path,
        "./data",
        "./database"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Global settings instance
settings = Settings()

# Create directories on import
create_directories()
