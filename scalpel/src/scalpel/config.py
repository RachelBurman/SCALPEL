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
    ollama_model: str = "qwen2.5:0.5b"
    
    # Model Parameters
    model_temperature: float = 0.3  # Lower for more analytical responses
    model_context_length: int = 32768  # Qwen 2.5 supports up to 128K
    
    # Paths
    papers_dir: Path = Path("data/papers")
    lancedb_path: Path = Path("data/lancedb")
    
    # Analysis Settings
    chunk_size: int = 2000  # Tokens per chunk for processing
    chunk_overlap: int = 200  # Overlap between chunks
    
    # Embedding Configuration
    embedding_model: str = "nomic-embed-text"  # Ollama embedding model
    embedding_dimensions: int = 768  # nomic-embed-text output dimensions
    embedding_batch_size: int = 32  # Batch size for embedding generation
    
    # LanceDB Settings
    lancedb_table: str = "scalpel_papers"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()