from typing import Optional
from cohere import ClientV2


class __CohereClient:
    """Lazy-initialized singleton-style wrapper for Cohere ClientV2 instances.

    This helper class defers creation of a ClientV2 until first access via `get()`
    and allows releasing the reference via `close()`.

    Attributes:
        __client: Cached ClientV2 instance or None if not initialized.
    """
    def __init__(self):
        self.__client: Optional[ClientV2] = None

    def get(self) -> ClientV2:
        if self.__client is None:
            self.__client = ClientV2()
        return self.__client

    def close(self):
        if self.__client is not None:
            self.__client = None


cohere_client = __CohereClient()
"""Singleton instance for the Cohere client."""
