"""
SCALPEL Configuration
Scientific Critique & Analysis Pipeline for Evidence Literature
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Provider selection: "ollama" (local) or "openrouter" (cloud)
    llm_provider: Literal["ollama", "openrouter"] = "ollama"

    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"

    # OpenRouter Configuration
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Model Parameters
    model_temperature: float = 0.3
    model_context_length: int = 32768

    # Paths
    papers_dir: Path = Path("data/papers")
    lancedb_path: Path = Path("data/lancedb")

    # Analysis Settings
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Embedding Configuration (always uses Ollama for local embeddings)
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 32

    # LanceDB Settings
    lancedb_table: str = "scalpel_papers"

    @property
    def active_model(self) -> str:
        """The model name for the active LLM provider."""
        if self.llm_provider == "openrouter":
            return self.openrouter_model
        return self.ollama_model

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
