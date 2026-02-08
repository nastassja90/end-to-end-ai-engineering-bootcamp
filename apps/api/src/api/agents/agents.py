from json import dumps
from typing import Generator, Optional
from langsmith import traceable
from langgraph.checkpoint.postgres import PostgresSaver

from api.server.models import RAGRequest
from api.core.config import config
from api.agents.rag.rag import used_context
from api.agents.tools.tools import get_formatted_context, get_formatted_reviews_context
from api.agents.internal.graph import init_workflow, process_graph_event
from api.utils.utils import get_tool_descriptions
from api.utils.tracing import hide_sensitive_inputs
from api.utils.streaming import string_for_sse


def __stream_agent(
    payload: RAGRequest, tools, initial_state
) -> Generator[str, None, dict]:
    """Internal generator that yields intermediate SSE chunks and returns the final result.
    Manages its own PostgreSQL connection to keep it alive during streaming."""
    result = None
    with PostgresSaver.from_conn_string(
        config.POSTGRES_CONNECTION_STRING
    ) as checkpointer:
        workflow = init_workflow(payload, tools)
        graph = workflow.compile(checkpointer=checkpointer)
        conf = {"configurable": {"thread_id": payload.thread_id}}

        for chunk in graph.stream(
            initial_state,
            config=conf,
            # Only include debug and values chunks in the stream, as those are the ones we want to process for intermediate updates and the final result.
            # debug: includes node start events that we use to generate intermediate status updates for the user
            # values: includes the final result of the graph execution that we want to return at the end of the stream.
            stream_mode=["debug", "values"],
        ):
            processed_chunk = process_graph_event(chunk)

            if processed_chunk:
                yield string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    return result


#### Agent Execution Function


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="run_agent",
)
def run_agent(
    payload: RAGRequest, stream: bool = False
) -> dict | Generator[str, None, dict]:

    tools = [get_formatted_context, get_formatted_reviews_context]
    tool_descriptions = get_tool_descriptions(tools)

    initial_state = {
        "messages": [{"role": "user", "content": payload.query}],
        "iteration": 0,
        "available_tools": tool_descriptions,
        # Add top_k to the initial state only if it's provided in extra_options
        **({"top_k": payload.extra_options.top_k} if payload.extra_options else {}),
    }

    if stream:
        # if stream is true, return a generator that manages its own connection.
        # use a separate function so that the generator is only returned in case of streaming, otherwise we can return the final result directly without the overhead of managing a generator and connection.
        return __stream_agent(payload, tools, initial_state)

    with PostgresSaver.from_conn_string(
        config.POSTGRES_CONNECTION_STRING
    ) as checkpointer:
        workflow = init_workflow(payload, tools)
        graph = workflow.compile(checkpointer=checkpointer)

        conf = {"configurable": {"thread_id": payload.thread_id}}
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

    gen = run_agent(payload, stream=True)
    result: Optional[dict] = None

    try:
        while True:
            chunk = next(gen)
            yield chunk  # propagate the chunk to the SSE stream
    except StopIteration as e:
        result = e.value  # the final result returned by the generator after completion

    yield string_for_sse(
        dumps(
            {
                "type": "final_result",
                "data": {
                    "answer": result.get("answer", ""),
                    "used_context": [
                        item.model_dump() for item in used_context(result)
                    ],
                    "trace_id": result.get("trace_id", ""),
                },
            }
        )
    )
