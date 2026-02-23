from json import dumps
from itertools import chain
from typing import Callable, Generator, List
from langsmith import traceable
from langgraph.checkpoint.postgres import PostgresSaver

from api.server.models import RAGRequest
from api.core.config import config
from api.agents.rag.rag import used_context
from api.agents.tools.tools import (
    get_formatted_item_context,
    get_formatted_reviews_context,
    add_to_shopping_cart,
    get_shopping_cart,
    remove_from_cart,
    check_warehouse_availability,
    reserve_warehouse_items,
)
from api.agents.basic.graph import (
    init_workflow as basic_workflow,
    process_graph_event as basic_process_graph_event,
)
from api.agents.advanced.graph import (
    init_workflow as advanced_workflow,
    process_graph_event as advanced_process_graph_event,
)
from api.utils.utils import get_tool_descriptions
from api.utils.tracing import hide_sensitive_inputs
from api.utils.streaming import string_for_sse

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_tools = [get_formatted_item_context, get_formatted_reviews_context]
"""List of tools to be used by the RAG agent."""

shopping_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
"""List of tools to be used by the shopping cart agent in the multi-agent workflow."""

warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]
"""List of tools to be used by the warehouse manager agent in the multi-agent workflow."""


def __initial_state(payload: RAGRequest, tools: List[List[Callable]]) -> dict:
    initial_state = {
        "messages": [{"role": "user", "content": payload.query}],
        "iteration": 0,
        "available_tools": get_tool_descriptions(list(chain(*tools))),
        # Add top_k to the initial state only if it's provided in extra_options
        **({"top_k": payload.extra_options.top_k} if payload.extra_options else {}),
    }
    if payload.execution_type == "multi-agent":
        initial_state = {
            "messages": [{"role": "user", "content": payload.query}],
            "user_id": payload.thread_id,
            "cart_id": payload.thread_id,
            "product_qa_agent": {
                "iteration": 0,
                "final_answer": False,
                "available_tools": get_tool_descriptions(rag_tools),
                "tool_calls": [],
                # Add top_k to the initial state only if it's provided in extra_options
                **(
                    {"top_k": payload.extra_options.top_k}
                    if payload.extra_options
                    else {}
                ),
            },
            "shopping_cart_agent": {
                "iteration": 0,
                "final_answer": False,
                "available_tools": get_tool_descriptions(shopping_tools),
                "tool_calls": [],
            },
            "warehouse_manager_agent": {
                "iteration": 0,
                "final_answer": False,
                "available_tools": get_tool_descriptions(warehouse_manager_agent_tools),
                "tool_calls": [],
            },
            "coordinator_agent": {
                "iteration": 0,
                "final_answer": False,
                "plan": [],
                "next_agent": "",
            },
        }
    return initial_state


def __get_shopping_cart_items(user_id: str, cart_id: str) -> List[dict]:
    logger.info(f"Retrieving shopping cart for user_id: {user_id}, cart_id: {cart_id}")
    shopping_cart = get_shopping_cart(user_id, cart_id)
    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": (
                float(item.get("total_price")) if item.get("total_price") else None
            ),
        }
        for item in shopping_cart
    ]
    return shopping_cart_items


def __stream_agent(
    payload: RAGRequest, tools: List[List[Callable]], initial_state
) -> Generator[str, None, dict]:
    """Internal generator that yields intermediate SSE chunks and returns the final result.
    Manages its own PostgreSQL connection to keep it alive during streaming."""

    try:

        result = None
        with PostgresSaver.from_conn_string(
            config.POSTGRES_CONNECTION_STRING
        ) as checkpointer:

            executor = basic_workflow
            if payload.execution_type == "multi-agent":
                executor = advanced_workflow

            workflow = executor(payload, tools)
            graph = workflow.compile(checkpointer=checkpointer)
            # max_concurrency=1: forces the ToolNode's ThreadPoolExecutor to run tools
            # sequentially, preventing the race condition in qdrant-fastembed's internal
            # batch accumulator dict (RuntimeError: dictionary changed size during iteration).
            conf = {
                "configurable": {"thread_id": payload.thread_id},
                "max_concurrency": 1,
            }

            for chunk in graph.stream(
                initial_state,
                config=conf,
                # Only include debug and values chunks in the stream, as those are the ones we want to process for intermediate updates and the final result.
                # debug: includes node start events that we use to generate intermediate status updates for the user
                # values: includes the final result of the graph execution that we want to return at the end of the stream.
                stream_mode=["debug", "values"],
            ):
                chunk_processor = basic_process_graph_event
                if payload.execution_type == "multi-agent":
                    chunk_processor = advanced_process_graph_event

                processed_chunk = chunk_processor(chunk)

                if processed_chunk:
                    yield string_for_sse(processed_chunk)

                if chunk[0] == "values":
                    result = chunk[1]

        # Emit the final_result chunk here, inside __stream_agent, so it is yielded
        # as a regular SSE chunk and not lost via StopIteration.value. The @traceable
        # decorator on run_agent wraps the generator with yield from, which correctly
        # forwards all yielded items but drops the return value (sets it to None).
        # Relying on StopIteration.value in rag_agent_stream would therefore always
        # receive None and crash before __get_shopping_cart_items is ever called.
        logger.info(f"Graph execution complete. Building final_result chunk.")
        context = [item.model_dump() for item in used_context(result)] if result else []
        shopping_cart = __get_shopping_cart_items(payload.thread_id, payload.thread_id)
        yield string_for_sse(
            dumps(
                {
                    "type": "final_result",
                    "data": {
                        "answer": result.get("answer", "") if result else "",
                        "used_context": context,
                        "trace_id": result.get("trace_id", "") if result else "",
                        "shopping_cart": shopping_cart,
                    },
                }
            )
        )
        return result
    except Exception as e:
        logger.exception(f"Error in __stream_agent: {e}")
        raise e


#### Agent Execution Function


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="run_agent",
)
def run_agent(payload: RAGRequest) -> dict | Generator[str, None, dict]:

    tools: List[List[Callable]] = (
        [rag_tools, shopping_tools, warehouse_manager_agent_tools]
        if payload.execution_type == "multi-agent"
        else [rag_tools]
    )

    initial_state = __initial_state(payload, tools)

    with PostgresSaver.from_conn_string(
        config.POSTGRES_CONNECTION_STRING
    ) as checkpointer:
        executor = basic_workflow
        if payload.execution_type == "multi-agent":
            executor = advanced_workflow
        workflow = executor(payload, tools)
        graph = workflow.compile(checkpointer=checkpointer)

        conf = {"configurable": {"thread_id": payload.thread_id}, "max_concurrency": 1}
        return graph.invoke(initial_state, config=conf)


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="rag_agent",
)
def rag_agent(
    payload: RAGRequest,
):
    result: dict = run_agent(payload)
    return {
        "answer": result.get("answer", ""),
        "used_context": used_context(result),
        "trace_id": result.get("trace_id", ""),
    }


def rag_agent_stream(
    payload: RAGRequest,
):
    tools: List[List[Callable]] = (
        [rag_tools, shopping_tools, warehouse_manager_agent_tools]
        if payload.execution_type == "multi-agent"
        else [rag_tools]
    )

    initial_state = __initial_state(payload, tools)
    # The final_result chunk (including shopping_cart) is emitted by __stream_agent
    # as the last yielded item before it returns. We simply forward all chunks here.
    # We do not rely on StopIteration.value because @traceable on run_agent wraps
    # the generator and loses the return value (always None).

    # in stream mode, return a generator that manages its own connection.
    # use a separate function so that the generator is only returned in case of streaming, otherwise we can return the final result directly without the overhead of managing a generator and connection.
    gen = __stream_agent(payload, tools, initial_state)

    try:
        while True:
            chunk = next(gen)
            yield chunk
    except StopIteration:
        pass
