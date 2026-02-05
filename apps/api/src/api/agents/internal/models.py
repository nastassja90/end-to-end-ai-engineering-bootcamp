from typing import Annotated, Any, Dict, List, Optional, TypeAlias
from pydantic import BaseModel, Field
from operator import add
from api.core.config import DEFAULT_TOP_K


class __ToolCallGetFormattedContextArguments(BaseModel):
    """Explicit arguments structure for get_formatted_context tool calls. Gemini requires explicit field definitions."""

    query: str = Field(
        ...,
        description="The search query to find relevant products. Extract keywords from the user's question.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description="The number of results to retrieve.",
    )


class __ToolCallGetFormattedReviewsContextArguments(BaseModel):
    """Explicit arguments structure for get_formatted_reviews_context tool calls. Gemini requires explicit field definitions."""

    query: str = Field(
        ...,
        description="The search query to find relevant reviews. Extract keywords from the user's question.",
    )
    item_list: List[str] = Field(
        ...,
        description="The list of item IDs to prefilter for before running the query.",
    )
    top_k: int = Field(
        default=15,
        description="The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered.",
    )


ToolCallArguments: TypeAlias = (
    __ToolCallGetFormattedContextArguments
    | __ToolCallGetFormattedReviewsContextArguments
)
"""Union type for tool call arguments."""


class ToolCall(BaseModel):
    """Represents a tool invocation with a strict tool name and its arguments.

    Attributes:
        name: Exact tool name to invoke (e.g., 'get_formatted_context').
        arguments: Arguments passed to the tool, including a required 'query'.
    """

    name: str = Field(
        ...,
        description="The exact name of the tool to call. Must be 'get_formatted_context', 'get_formatted_reviews_context', etc...",
    )
    arguments: ToolCallArguments = Field(
        ...,
        description="The arguments to pass to the tool. Must include 'query' with search keywords, along with any other required parameters.",
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
    top_k: int = DEFAULT_TOP_K
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[ReferencedItem], add] = []
    trace_id: str = ""
