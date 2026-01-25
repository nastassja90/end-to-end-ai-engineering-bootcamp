from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    # Default RAG prompt key to customize the prompt used in the RAG pipeline
    RAG_PROMPT_KEY: str = "retrieval_generation"
    # Langsmith environment variables
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    LANGSMITH_API_KEY: str
    # Cohere API Key for re-ranking models
    CO_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")


config = Config()
