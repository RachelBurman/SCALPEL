"""
SCALPEL Configuration
Scientific Critique & Analysis Pipeline for Evidence Literature
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"
    
    # Model Parameters
    model_temperature: float = 0.3  # Lower for more analytical responses
    model_context_length: int = 32768  # Qwen 2.5 supports up to 128K
    
    # Paths
    papers_dir: Path = Path("data/papers")
    
    # Analysis Settings
    chunk_size: int = 2000  # Tokens per chunk for processing
    chunk_overlap: int = 200  # Overlap between chunks
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure papers directory exists
        self.papers_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()