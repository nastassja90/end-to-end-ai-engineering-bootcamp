from fastapi import Request, APIRouter
from api.server.models import (
    RAGRequest,
    RAGResponse,
    ConfigResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from api.core.config import config
from api.core.constants import MODELS, OPENAI, GROQ, GOOGLE
from api.agents.rag.rag import rag_pipeline
from api.agents.agents import rag_agent
from api.server.processors.feedback import submit_feedback

import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    )

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_context=result["used_context"],
        trace_id=result["trace_id"],
    )


config_router = APIRouter()


@config_router.get("/")
def get_config() -> ConfigResponse:
    """Return the application configuration including available models and providers."""
    return ConfigResponse(
        models=MODELS,
        providers=[OPENAI, GROQ, GOOGLE],
    )


feedback_router = APIRouter()


@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:

    submit_feedback(
        payload.trace_id,
        payload.feedback_score,
        payload.feedback_text,
        payload.feedback_source_type,
    )

    return FeedbackResponse(request_id=request.state.request_id, status="success")


api_router = APIRouter()
api_router.include_router(config_router, prefix="/config", tags=["config"])
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(feedback_router, prefix="/feedback", tags=["feedback"])
