from functools import partial
from typing import Callable, List

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from api.agents.advanced.nodes import (
    coordinator_agent,
    product_qa_agent,
    shopping_cart_agent,
    warehouse_manager_agent,
)
from api.agents.common.models import StateAdvanced as State, ToolCall
from api.server.models import RAGRequest


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

    def _is_node_start(chunk):
        return chunk[1].get("type") == "task"

    def _is_node_end(chunk):
        return chunk[0] == "updates"

    def _tool_to_text(tool_call: ToolCall):
        if tool_call.name == "get_formatted_item_context":
            return f"Looking for items: {tool_call.arguments.query}."
        elif tool_call.name == "get_formatted_reviews_context":
            return f"Fetching user reviews..."
        elif tool_call.name == "add_to_shopping_cart":
            return f"Adding {tool_call.arguments.items} to the shopping cart."
        elif tool_call.name == "get_shopping_cart":
            return f"Retrieving the shopping cart."
        elif tool_call.name == "remove_from_cart":
            return f"Removing {tool_call.arguments.items} from the shopping cart."
        elif tool_call.name == "check_warehouse_availability":
            return f"Checking warehouse availability for {tool_call.arguments.items}."
        elif tool_call.name == "reserve_warehouse_items":
            return f"Reserving warehouse items: {tool_call.arguments.reservations}."
        else:
            return f"Unknown tool: {tool_call.name}."

    if _is_node_start(chunk):
        if chunk[1].get("payload", {}).get("name") == "coordinator_agent":
            return "Analyzing the question..."
        if chunk[1].get("payload", {}).get("name") == "product_qa_agent":
            return "Planning..."
        if chunk[1].get("payload", {}).get("name") == "shopping_cart_agent":
            return "Managing the shopping cart..."
        if chunk[1].get("payload", {}).get("name") == "warehouse_manager_agent":
            return "Managing the warehouse..."
        if chunk[1].get("payload", {}).get("name", "").endswith("_tool_node"):
            node_name = chunk[1].get("payload", {}).get("name", "")
            input_data = chunk[1].get("payload", {}).get("input", {})

            # input_data is a StateAdvanced pydantic object, access the right sub-agent
            if node_name == "product_qa_agent_tool_node":
                tool_calls = (
                    input_data.product_qa_agent.tool_calls
                    if hasattr(input_data, "product_qa_agent")
                    else []
                )
            elif node_name == "shopping_cart_agent_tool_node":
                tool_calls = (
                    input_data.shopping_cart_agent.tool_calls
                    if hasattr(input_data, "shopping_cart_agent")
                    else []
                )
            elif node_name == "warehouse_manager_agent_tool_node":
                tool_calls = (
                    input_data.warehouse_manager_agent.tool_calls
                    if hasattr(input_data, "warehouse_manager_agent")
                    else []
                )
            else:
                tool_calls = []

            message = " ".join([_tool_to_text(tool_call) for tool_call in tool_calls])
            return message
    else:
        return False


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

    if state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        # tool_calls takes priority over final_answer: same reasoning as above.
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
def init_workflow(payload: RAGRequest, tools: List[List[Callable]]) -> StateGraph:

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
        {"tools": "shopping_cart_agent_tool_node", "end": "coordinator_agent"},
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
