from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from api.core.config import DEFAULT_TOP_K, MAX_TOP_K


ExecutionType = Literal["pipeline", "agent"]
"""Enumeration for the type of RAG execution."""


class RAGRequestExtraOptions(BaseModel):
    top_k: int = Field(
        DEFAULT_TOP_K,
        description="The number of top relevant documents to retrieve",
    )
    enable_reranking: bool = Field(
        False, description="Whether to enable re-ranking (default: False)"
    )


class RAGRequest(BaseModel):
    execution_type: ExecutionType = Field(
        "pipeline",
        description="The type of RAG execution: 'pipeline' or 'agent'. Default is 'pipeline'.",
    )
    provider: str = Field(..., description="The LLM provider to use")
    model_name: str = Field(..., description="The model name to use")
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: str = Field(..., description="The thread ID for the conversation")
    extra_options: Optional[RAGRequestExtraOptions] = Field(
        None,
        description="Additional options for the RAG pipeline",
    )


class RAGUsedContextItem(BaseModel):
    image_url: Optional[str] = Field(
        None, description="The image URL of the referenced item"
    )
    price: Optional[float] = Field(None, description="The price of the referenced item")
    description: str = Field(..., description="The description of the referenced item")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContextItem] = Field(
        default=[],
        description="The contextual info for each item used to generate the answer",
    )
    trace_id: str = Field("", description="The trace ID for the request")


class ConfigResponse(BaseModel):
    models: dict[str, list[str]] = Field(
        ..., description="Dictionary of providers and their available models"
    )
    providers: list[str] = Field(..., description="List of available providers")
    top_k: dict[str, int] = Field(
        {
            "default": DEFAULT_TOP_K,
            "max": MAX_TOP_K,
        },
        description="Top k configuration",
    )


class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(
        ..., description="1 if the feedback is positive, 0 if the feedback is negative"
    )
    feedback_text: str = Field(..., description="The feedback text")
    trace_id: str = Field(..., description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")
    feedback_source_type: str = Field(
        ..., description="The type of feedback. Human or API"
    )


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")
