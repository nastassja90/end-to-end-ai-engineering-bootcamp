from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration settings for the API application."""

    # Logging configuration
    LOG_LEVEL: str = "INFO"

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 10001

    # API Keys for AI providers
    OPENAI_API_KEY: str
    POSTGRES_CONNECTION_STRING: str = "postgresql://user:pwd@localhost:5432/db"
    """Postgres connection string used by LangGraph to store conversation state in multi-turn agent conversation"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config: Config = Config()
"""Singleton instance of the application configuration loaded from environment variables."""
