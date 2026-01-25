from pydantic import BaseModel, Field
from typing import Optional


class RAGRequestExtraOptions(BaseModel):
    top_k: int = Field(
        5, description="The number of top relevant documents to retrieve (default: 5)"
    )
    enable_reranking: bool = Field(
        False, description="Whether to enable re-ranking (default: False)"
    )


class RAGRequest(BaseModel):
    provider: str = Field(..., description="The LLM provider to use")
    model_name: str = Field(..., description="The model name to use")
    query: str = Field(..., description="The query to be used in the RAG pipeline")
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
