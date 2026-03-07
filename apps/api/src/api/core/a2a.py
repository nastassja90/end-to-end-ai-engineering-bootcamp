from typing import Optional
from a2a.client import A2ACardResolver
from a2a.types import AgentCard

from httpx import Timeout, AsyncClient


class A2AClient:
    def __init__(self):
        self.__timeout_config = Timeout(
            connect=10.0,  # Connection timeout
            read=100.0,  # Read timeout (important for long-running operations)
            write=10.0,  # Write timeout
            pool=10.0,  # Pool timeout
        )
        self.__httpx_client: Optional[AsyncClient] = None

    async def __open_httpx_client(self):
        if self.__httpx_client is None:
            self.__httpx_client = await AsyncClient(timeout=self.__timeout_config)

    async def get_agent_card(self, agent_url: str) -> AgentCard:

        async with self.__open_httpx_client() as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_url)
            agent_card = await resolver.get_agent_card()
            return agent_card

    async def close_httpx_client(self):
        if self.__httpx_client is not None:
            await self.__httpx_client.aclose()
            self.__httpx_client = None
