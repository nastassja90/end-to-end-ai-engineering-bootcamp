from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration settings for the API application."""

    # API Keys for AI providers
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    CO_API_KEY: str
    # Qdrant URL for vector database connection
    QDRANT_URL: str = "http://localhost:6333"
    # Langsmith environment variables
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    LANGSMITH_API_KEY: str
    # Default RAG prompt key to customize the prompt used in the RAG pipeline
    RAG_PROMPT_KEY: str = "retrieval_generation"
    # Qdrant collection name for RAG operations
    RAG_COLLECTION_NAME: str = "Amazon-items-collection-01-hybrid-search"
    # The text embedding model to use for both indexing and querying in the RAG pipeline
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # The reranking model to use in the RAG pipeline
    RAG_RERANKING_MODEL: str = "rerank-v4.0-fast"

    model_config = SettingsConfigDict(env_file=".env")
