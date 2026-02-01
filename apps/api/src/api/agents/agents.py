from qdrant_client import QdrantClient
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue
from api.server.models import RAGRequest
from api.core.config import Config
from api.agents.tools.tools import get_formatted_context
from api.agents.internal.graph import init_workflow
from api.utils.utils import get_tool_descriptions
from langsmith import traceable
from api.utils.tracing import hide_sensitive_inputs


#### Agent Execution Function


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="run_agent",
)
def run_agent(
    app_config: Config,
    payload: RAGRequest,
) -> dict:

    tools = [get_formatted_context]
    tool_descriptions = get_tool_descriptions(tools)

    initial_state = {
        "messages": [{"role": "user", "content": payload.query}],
        "iteration": 0,
        "available_tools": tool_descriptions,
    }

    workflow = init_workflow(app_config, payload, tools)
    graph = workflow.compile()
    return graph.invoke(initial_state)


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="rag_agent",
)
def rag_agent(
    app_config: Config,
    payload: RAGRequest,
    qdrant_client: QdrantClient,
):

    result = run_agent(app_config, payload)
    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = (
            qdrant_client.query_points(
                collection_name=app_config.RAG_COLLECTION_NAME,
                query=dummy_vector,
                limit=1,
                using=app_config.RAG_EMBEDDING_MODEL,
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
    }
