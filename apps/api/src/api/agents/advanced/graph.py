from functools import partial
from json import dumps
from typing import Callable, List, Union

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from api.agents.advanced.nodes import (
    coordinator_agent,
    product_qa_agent,
    shopping_cart_agent,
    warehouse_manager_agent,
    hitl_add_to_cart,
)
from api.agents.common.models import StateAdvanced as State, ToolCall
from api.server.models import HitlRequest, RAGRequest


def process_graph_event(chunk):
    """Process a LangGraph event chunk and return a human-readable status message.

    Interprets the event structure to detect node starts and specific node types
    (intent router, agent, tool) and formats tool calls into user-facing text.
    Returns False when no relevant message is produced.

    Args:
        chunk: A sequence/tuple-like event containing metadata and payload data.

    Returns:
        str or bool: A status message for recognized node starts, or False otherwise.
    """

    # identify if the chunk is a LangGraph interrupt event, which is
    # used for HITL interactions.
    def _is_interrupt(chunk):
        return len(chunk[1].get("payload", {}).get("interrupts", [])) > 0

    def _is_node_start(chunk):
        return chunk[1].get("type") == "task"

    def _is_node_end(chunk):
        return chunk[0] == "updates"

    def _tool_to_text(tool_call):
        if tool_call.name == "get_formatted_items_context":
            return f"Looking for items: {tool_call.arguments.get('query', '')}."
        elif tool_call.name == "get_formatted_reviews_context":
            return f"Fetching user reviews..."
        else:
            return f"Unknown tool: {tool_call.name}."

    if _is_node_start(chunk):
        if chunk[1].get("payload", {}).get("name") == "coordinator_agent":
            return "Planning..."
        if chunk[1].get("payload", {}).get("name") == "product_qa_agent":
            return "Fetching information about inventory..."
        if chunk[1].get("payload", {}).get("name") == "shopping_cart_agent":
            return "Shopping cart management..."
        if chunk[1].get("payload", {}).get("name") == "warehouse_manager_agent":
            return "Warehouse management..."
        if chunk[1].get("payload", {}).get("name") == "tool_node":
            message = " ".join(
                [
                    _tool_to_text(tool_call)
                    for tool_call in chunk[1]
                    .get("payload", {})
                    .get("input", {})
                    .tool_calls
                ]
            )
            return message
    elif _is_interrupt(chunk):
        value = chunk[1].get("payload", {}).get("interrupts", [])[0].get("value")
        payload = {"type": "hitl_interrupt", "data": {"data": value}}
        return dumps(payload)
    else:
        return "Unknown operation..."


#### Edges


def product_qa_agent_tool_edge(state: State) -> str:
    """Decide whether to continue or end"""

    if state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        # tool_calls takes priority over final_answer: if the LLM emits both
        # simultaneously (violating prompt rules), we must still execute the
        # requested tools. Skipping them would leave an orphaned AIMessage with
        # tool_calls in state.messages, causing a 400 on the next OpenAI call.
        return "tools"
    elif state.product_qa_agent.final_answer:
        return "end"
    else:
        return "end"


def shopping_cart_agent_tool_edge(state: State) -> str:
    """Decide whether to continue or end"""

    add_to_cart_tool_call = False
    for tool_call in state.shopping_cart_agent.tool_calls:
        if tool_call.name == "add_to_shopping_cart":
            add_to_cart_tool_call = True
            break

    if state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        # tool_calls takes priority over final_answer: same reasoning as above.
        if add_to_cart_tool_call:
            return "hitl_add_to_cart"
        else:
            return "tools"
    elif state.shopping_cart_agent.final_answer:
        return "end"
    else:
        return "end"


def warehouse_manager_agent_tool_edge(state: State) -> str:
    """Decide whether to continue or end"""

    if state.warehouse_manager_agent.final_answer:
        return "end"
    elif state.warehouse_manager_agent.iteration > 2:
        return "end"
    elif len(state.warehouse_manager_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def coordinator_agent_edge(state: State) -> str:

    if state.coordinator_agent.iteration > 3:
        return "end"
    elif (
        state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0
    ):
        return "end"
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.coordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return "end"


#### Workflow
def init_workflow(
    payload: Union[RAGRequest, HitlRequest], tools: List[List[Callable]]
) -> StateGraph:

    workflow = StateGraph(State)

    product_qa_agent_tools, shopping_cart_agent_tools, warehouse_manager_agent_tools = (
        tools
    )

    product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)
    shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
    warehouse_manager_agent_tool_node = ToolNode(warehouse_manager_agent_tools)

    workflow.add_node(
        "product_qa_agent",
        partial(
            product_qa_agent,
            provider=payload.provider,
            model_name=payload.model_name,
        ),
    )
    workflow.add_node(
        "shopping_cart_agent",
        partial(
            shopping_cart_agent,
            provider=payload.provider,
            model_name=payload.model_name,
        ),
    )
    workflow.add_node(
        "warehouse_manager_agent",
        partial(
            warehouse_manager_agent,
            provider=payload.provider,
            model_name=payload.model_name,
        ),
    )
    workflow.add_node(
        "coordinator_agent",
        partial(
            coordinator_agent,
            provider=payload.provider,
            model_name=payload.model_name,
        ),
    )

    workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
    workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)
    workflow.add_node(
        "warehouse_manager_agent_tool_node", warehouse_manager_agent_tool_node
    )
    workflow.add_node("hitl_add_to_cart", hitl_add_to_cart)

    workflow.add_edge(START, "coordinator_agent")

    workflow.add_conditional_edges(
        "coordinator_agent",
        coordinator_agent_edge,
        {
            "product_qa_agent": "product_qa_agent",
            "shopping_cart_agent": "shopping_cart_agent",
            "warehouse_manager_agent": "warehouse_manager_agent",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "product_qa_agent",
        product_qa_agent_tool_edge,
        {"tools": "product_qa_agent_tool_node", "end": "coordinator_agent"},
    )

    workflow.add_conditional_edges(
        "shopping_cart_agent",
        shopping_cart_agent_tool_edge,
        {
            "tools": "shopping_cart_agent_tool_node",
            "hitl_add_to_cart": "hitl_add_to_cart",
            "end": "coordinator_agent",
        },
    )

    workflow.add_conditional_edges(
        "warehouse_manager_agent",
        warehouse_manager_agent_tool_edge,
        {"tools": "warehouse_manager_agent_tool_node", "end": "coordinator_agent"},
    )

    workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
    workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")
    workflow.add_edge("warehouse_manager_agent_tool_node", "warehouse_manager_agent")
    return workflow
