from typing import Optional
from qdrant_client import QdrantClient
from api.core.config import config, Env


class __QdrantClient:
    """Manage a lazily initialized, singleton-style QdrantClient instance.

    This helper stores the Qdrant URL and API key, creates the client on first
    access, and provides a method to close and reset the client to release
    resources when no longer needed."""

    def __init__(self, env: Env, url: str, api_key: str):
        self.__env: Env = env
        self.__url: str = url
        self.__api_key: str = api_key
        self.__client: Optional[QdrantClient] = None

    def get(self) -> QdrantClient:
        """Return a singleton QdrantClient instance, lazily initializing it with URL and API key on first access."""

        if self.__client is None:
            if self.__env == Env.DEV:
                print(f"Initializing Qdrant client with URL: {self.__url}")
                self.__client = QdrantClient(url=self.__url)
            else:
                self.__client = QdrantClient(url=self.__url, api_key=self.__api_key)

        return self.__client

    def close(self):
        """Close the underlying client connection if it exists and reset the client.

        This method safely shuts down the internal client instance (if initialized)
        and sets it to None to release resources.
        """
        if self.__client is not None:
            self.__client.close()
            self.__client = None


qdrant_client = __QdrantClient(
    env=config.ENV,
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
)
"""Singleton instance of the Qdrant client manager."""
