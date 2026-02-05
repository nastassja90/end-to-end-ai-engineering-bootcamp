import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue
from api.server.models import RAGRequest
from api.core.config import Config, RAG_COLLECTIONS, RAG_EMBEDDING_MODEL
from api.core.qdrant import qdrant_client
from api.agents.tools.tools import get_formatted_context, get_formatted_reviews_context
from api.agents.internal.graph import init_workflow
from api.utils.utils import get_tool_descriptions
from langsmith import traceable
from api.utils.tracing import hide_sensitive_inputs
from langgraph.checkpoint.postgres import PostgresSaver


#### Agent Execution Function


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="run_agent",
)
def run_agent(
    app_config: Config,
    payload: RAGRequest,
) -> dict:

    tools = [get_formatted_context, get_formatted_reviews_context]
    tool_descriptions = get_tool_descriptions(tools)

    initial_state = {
        "messages": [{"role": "user", "content": payload.query}],
        "iteration": 0,
        "available_tools": tool_descriptions,
        # Add top_k to the initial state only if it's provided in extra_options
        **({"top_k": payload.extra_options.top_k} if payload.extra_options else {}),
    }

    config = {"configurable": {"thread_id": payload.thread_id}}

    with PostgresSaver.from_conn_string(
        app_config.POSTGRES_CONNECTION_STRING
    ) as checkpointer:
        workflow = init_workflow(app_config, payload, tools)
        graph = workflow.compile(checkpointer=checkpointer)
        return graph.invoke(initial_state, config=config)


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="rag_agent",
)
def rag_agent(
    app_config: Config,
    payload: RAGRequest,
):

    result = run_agent(app_config, payload)
    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = (
            qdrant_client.get()
            .query_points(
                collection_name=RAG_COLLECTIONS["items"],
                query=dummy_vector,
                limit=1,
                using=RAG_EMBEDDING_MODEL,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin", match=MatchValue(value=item.id)
                        )
                    ]
                ),
            )
            .points[0]
            .payload
        )
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.description,
                }
            )

    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
        "trace_id": result.get("trace_id", ""),
    }
