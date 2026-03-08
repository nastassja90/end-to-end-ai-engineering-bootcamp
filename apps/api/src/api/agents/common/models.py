from typing import Annotated, Any, Dict, List, Optional, TypeAlias
from pydantic import BaseModel, Field
from operator import add
from api.core.config import DEFAULT_TOP_K
from langgraph.graph.message import add_messages


class __ToolCallGetFormattedContextArguments(BaseModel):
    """Explicit arguments structure for get_formatted_item_context tool calls. Gemini requires explicit field definitions."""

    query: str = Field(
        ...,
        description="The search query to find relevant products. Extract keywords from the user's question.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description="The number of results to retrieve.",
    )
    enable_reranking: bool = Field(
        default=False,
        description="Whether to enable reranking of retrieved results.",
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


class __ToolCallAddToShoppingCartArguments(BaseModel):
    """Arguments structure for add_to_shopping_cart tool calls."""

    items: List[Dict[str, Any]] = Field(
        ...,
        description="A list of items to add to the shopping cart. Each item must have 'product_id' (str) and 'quantity' (int).",
    )
    user_id: str = Field(
        ...,
        description="The id of the user to add the items to the shopping cart.",
    )
    cart_id: str = Field(
        ...,
        description="The id of the shopping cart to add the items to.",
    )


class __ToolCallGetShoppingCartArguments(BaseModel):
    """Arguments structure for get_shopping_cart tool calls."""

    user_id: str = Field(
        ...,
        description="The id of the user whose shopping cart to retrieve.",
    )
    cart_id: str = Field(
        ...,
        description="The id of the shopping cart to retrieve.",
    )


class __ToolCallRemoveFromCartArguments(BaseModel):
    """Arguments structure for remove_from_cart tool calls."""

    product_id: str = Field(
        ...,
        description="The product ID to remove from the shopping cart.",
    )
    user_id: str = Field(
        ...,
        description="The id of the user whose cart to modify.",
    )
    cart_id: str = Field(
        ...,
        description="The id of the shopping cart to remove the item from.",
    )


class __ToolCallCheckWarehouseAvailabilityArguments(BaseModel):
    """Arguments structure for check_warehouse_availability tool calls."""

    items: List[Dict[str, Any]] = Field(
        ...,
        description="The list of products to check availability for.",
    )


class __ToolCallReserveWarehouseItemsArguments(BaseModel):
    """Arguments structure for reserve_warehouse_items tool calls."""

    reservations: List[Dict[str, Any]] = Field(
        ...,
        description="The list of reservations to make in the warehouse.",
    )


ToolCallArguments: TypeAlias = (
    __ToolCallGetFormattedReviewsContextArguments  # 3 fields (query, item_list, top_k) - more specific than GetFormattedContext
    | __ToolCallAddToShoppingCartArguments  # 3 fields (items, user_id, cart_id)
    | __ToolCallRemoveFromCartArguments  # 3 fields (product_id, user_id, cart_id)
    | __ToolCallGetFormattedContextArguments  # 3 fields (query, top_k, enable_reranking)
    | __ToolCallGetShoppingCartArguments  # 2 fields (user_id, cart_id) - least specific, goes last
    | __ToolCallCheckWarehouseAvailabilityArguments  # 1 field (items) - least specific, goes last
    | __ToolCallReserveWarehouseItemsArguments  # 1 field (reservations) - least specific, goes last
)
"""Union type for tool call arguments (RAG tools + shopping cart tools).
Ordered from most specific (most required fields) to least specific to ensure
correct Pydantic union discrimination."""


class ToolCall(BaseModel):
    """Represents a tool invocation with a strict tool name and its arguments.

    Attributes:
        name: Exact tool name to invoke (e.g., 'get_formatted_item_context').
        arguments: Arguments passed to the tool, including a required 'query'.
    """

    name: str = Field(
        ...,
        description="The exact name of the tool to call. Must be 'get_formatted_item_context', 'get_formatted_reviews_context', etc...",
        pattern=r"^[a-zA-Z0-9_-]+$",
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


### Product QnA Agent Response Model
class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(
        description="Short description of the item used to answer the question"
    )


class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(
        description="List of items used to answer the question."
    )
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Shopping cart Agent Response Model
class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


class Delegation(BaseModel):
    agent: str
    task: str


### Warehouse Manager Agent Response Model
class WarehouseManagerAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


### Coordinator Agent Response Model
class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: List[Delegation]
    final_answer: bool = False
    answer: str = ""


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
    enable_reranking: bool = False
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[ReferencedItem], add] = []
    trace_id: str = ""


class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []


class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""


class StateAdvanced(BaseModel):
    messages: Annotated[List[Any], add_messages] = []
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    warehouse_manager_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(
        default_factory=CoordinatorAgentProperties
    )
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""
    top_k: int = DEFAULT_TOP_K
    enable_reranking: bool = False
    trace_id: str = ""
