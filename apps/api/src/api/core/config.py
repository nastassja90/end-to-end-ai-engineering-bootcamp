from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, TypeAlias

OPENAI: str = "OpenAI"
GROQ: str = "Groq"
GOOGLE: str = "Google"

LLMProvider: TypeAlias = Literal[f"{OPENAI}", f"{GROQ}", f"{GOOGLE}"]
"""Type alias for supported LLM providers."""

MODELS: dict[LLMProvider, list[str]] = {
    OPENAI: ["gpt-4.1-mini", "gpt-4.1"],
    GROQ: ["groq/llama-3.3-70b-versatile"],
    # TODO: skip for now GOOGLE models because they are unstable with the integration of
    # Lite LLM Router and usage metadata extraction when using the gemini models supported by Lite LLM (gemini/gemini-2.0-flash)
    # GOOGLE: ["gemini-2.5-flash"],
}
"""Available AI providers and their respective models."""

DEFAULT_TOP_K: int = 5
"""Default value for the number of relevant documents to retrieve in the RAG pipeline."""
MAX_TOP_K: int = 20
"""Maximum value for the number of relevant documents to retrieve in the RAG pipeline."""

"""Default RAG prompt key to customize the prompt used in the RAG pipeline"""
RAG_COLLECTIONS: dict[str, str] = {
    "items": "Amazon-items-collection-01-hybrid-search",
    "reviews": "Amazon-items-collection-01-reviews",
}
"""Qdrant collection names for RAG operations"""
RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
"""The text embedding model to use for both indexing and querying in the RAG pipeline"""
RAG_RERANKING_MODEL: str = "rerank-v4.0-fast"
"""The reranking model to use in the RAG pipeline"""


class Env(str, Enum):
    DEV = "dev"
    PROD = "prod"


class Config(BaseSettings):
    """Configuration settings for the API application."""

    # Logging configuration
    LOG_LEVEL: str = "INFO"
    ENV: Env = Env.DEV
    """Application environment, either 'dev' or 'prod'. Determines environment-specific behavior and settings."""

    # API Keys for AI providers
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    CO_API_KEY: str
    QDRANT_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    """Qdrant URL for vector database connection"""
    POSTGRES_CONNECTION_STRING: str = "postgresql://user:pwd@localhost:5432/db"
    """Postgres connection string used by LangGraph to store conversation state in multi-turn agent conversation"""
    # Langsmith environment variables
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    LANGSMITH_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")


config: Config = Config()
"""Singleton instance of the application configuration loaded from environment variables."""
