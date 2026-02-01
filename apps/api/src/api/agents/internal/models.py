from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, Field
from operator import add


class ToolCallArguments(BaseModel):
    """Explicit arguments structure for tool calls. Gemini requires explicit field definitions."""

    query: str = Field(
        ...,
        description="The search query to find relevant products. Extract keywords from the user's question.",
    )
    top_k: int = Field(
        default=5,
        description="The number of results to retrieve. Default is 5.",
    )


class ToolCall(BaseModel):
    name: str = Field(
        ...,
        description="The exact name of the tool to call. Must be 'get_formatted_context'.",
    )
    arguments: ToolCallArguments = Field(
        ...,
        description="The arguments to pass to the tool. Must include 'query' with search keywords.",
    )


#############################################################
# Define the Structured Output response type for Instructor #
#############################################################


# Define the pydantic model for the ReferencedItem retrieved from the vector database.
class ReferencedItem(BaseModel):
    id: str = Field(
        ..., description="The unique identifier of the referenced item (parent ASIN)."
    )
    description: str = Field(
        ..., description="The short description of the referenced item."
    )


# Define the output schema using Pydantic. This schema will be used to structure the model's response via instructor.
class StructuredResponse(BaseModel):
    answer: str = Field(
        ..., description="A brief summary of the weather in Italy today."
    )
    references: list[ReferencedItem] = Field(
        ..., description="A list of items used to answer the question."
    )
    tool_calls: List[ToolCall] = []
    final_answer: bool = False


#############################################################


### Intent Router Response Model


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str


class RAGRetrievedContext(BaseModel):
    retrieved_context_ids: list[str] = list()
    retrieved_context: list[str] = list()
    similarity_scores: list[float] = list()
    retrieved_context_ratings: list[Optional[float]] = list()


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[ReferencedItem], add] = []
