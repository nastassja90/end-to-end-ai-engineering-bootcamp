from typing import Callable, List
from api.server.models import RAGRequest
from api.core.config import Config
from api.agents.internal.models import State
from api.agents.internal.nodes import agent_node, intent_router_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from functools import partial


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


def init_workflow(
    config: Config, payload: RAGRequest, tools: List[Callable]
) -> StateGraph:

    workflow = StateGraph(State)

    tool_node = ToolNode(tools)

    workflow.add_node(
        "agent_node",
        partial(
            agent_node,
            app_config=config,
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
