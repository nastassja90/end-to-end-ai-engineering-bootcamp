from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    provider: str = Field(..., description="The LLM provider to use")
    model_name: str = Field(..., description="The model name to use")
    query: str = Field(..., description="The query to be used in the RAG pipeline")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
