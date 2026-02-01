from fastapi import Request, APIRouter
from api.server.models import RAGRequest, RAGResponse, ConfigResponse
from api.core.config import Config
from api.core.constants import MODELS, OPENAI, GROQ, GOOGLE

from qdrant_client import QdrantClient
from api.agents.rag import rag_pipeline
from api.agents.agents import rag_agent

import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()

qdrant_client = QdrantClient(url=config.QDRANT_URL)

rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:
    logger.info(
        f"RAG request received. Request ID: {request.state.request_id}, Question: {payload.query}"
    )
    logger.info(f"Using prompt key: {config.RAG_PROMPT_KEY}")

    executor = rag_pipeline
    if payload.execution_type == "agent":
        executor = rag_agent

    result = executor(
        app_config=config,
        payload=payload,
        qdrant_client=qdrant_client,
    )

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_context=result["used_context"],
    )


config_router = APIRouter()


@config_router.get("/")
def get_config() -> ConfigResponse:
    """Return the application configuration including available models and providers."""
    return ConfigResponse(
        models=MODELS,
        providers=[OPENAI, GROQ, GOOGLE],
    )


api_router = APIRouter()
api_router.include_router(config_router, prefix="/config", tags=["config"])
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
