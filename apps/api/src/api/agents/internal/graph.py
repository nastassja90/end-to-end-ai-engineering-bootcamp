from typing import Callable, List
from api.server.models import RAGRequest
from api.agents.internal.models import State
from api.agents.internal.nodes import agent_node, intent_router_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from functools import partial


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

    def _tool_to_text(tool_call):
        if tool_call.name == "get_formatted_context":
            return f"Looking for items: {tool_call.arguments.get('query', '')}."
        elif tool_call.name == "get_formatted_reviews_context":
            return f"Fetching user reviews..."
        else:
            return f"Unknown tool: {tool_call.name}."

    if _is_node_start(chunk):
        if chunk[1].get("payload", {}).get("name") == "intent_router_node":
            return "Analyzing the question..."
        if chunk[1].get("payload", {}).get("name") == "agent_node":
            return "Planning..."
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
    else:
        return False


#### Edges


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""

    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def intent_router_conditional_edges(state: State):

    if state.question_relevant:
        return "agent_node"
    else:
        return "end"


#### Workflow


def init_workflow(payload: RAGRequest, tools: List[Callable]) -> StateGraph:

    workflow = StateGraph(State)

    tool_node = ToolNode(tools)

    workflow.add_node(
        "agent_node",
        partial(
            agent_node,
            provider=payload.provider,
            model_name=payload.model_name,
        ),
    )
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("intent_router_node", intent_router_node)

    workflow.add_edge(START, "intent_router_node")

    workflow.add_conditional_edges(
        "intent_router_node",
        intent_router_conditional_edges,
        {"agent_node": "agent_node", "end": END},
    )

    workflow.add_conditional_edges(
        "agent_node", tool_router, {"tools": "tool_node", "end": END}
    )

    workflow.add_edge("tool_node", "agent_node")
    return workflow
