from typing import Optional
from qdrant_client import QdrantClient
from api.core.config import config


class __QdrantClient:
    """Manage a lazily initialized, singleton-style QdrantClient instance.

    This helper stores the Qdrant URL, creates the client on first
    access, and provides a method to close and reset the client to release
    resources when no longer needed."""

    def __init__(self, url: str):
        self.__url: str = url
        self.__client: Optional[QdrantClient] = None

    def get(self) -> QdrantClient:
        """Return a singleton QdrantClient instance, lazily initializing it with URL on first access."""

        if self.__client is None:
            self.__client = QdrantClient(url=self.__url)
        return self.__client

    def close(self):
        """Close the underlying client connection if it exists and reset the client.

        This method safely shuts down the internal client instance (if initialized)
        and sets it to None to release resources.
        """
        if self.__client is not None:
            self.__client.close()
            self.__client = None


qdrant_client = __QdrantClient(url=config.QDRANT_URL)
"""Singleton instance of the Qdrant client manager."""
