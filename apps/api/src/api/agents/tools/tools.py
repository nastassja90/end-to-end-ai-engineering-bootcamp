from api.agents.rag import process_context, retrieve_data
from api.core.config import Config
from api.server.models import RAGRequestExtraOptions
from qdrant_client import QdrantClient
from api.utils.tracing import hide_sensitive_inputs
from langsmith import traceable


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="get_formatted_context",
    run_type="tool",
)
def get_formatted_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    # TODO: top_k and enable_reranking should come from extra_options in the original RAGRequest
    # Build config and extra options expected by retrieve_data
    app_config = Config()
    extra_options = RAGRequestExtraOptions(top_k=top_k, enable_reranking=False)

    qdrant_client = QdrantClient(url=app_config.QDRANT_URL)

    context = retrieve_data(query, qdrant_client, extra_options)
    formatted_context = process_context(context)

    return formatted_context
