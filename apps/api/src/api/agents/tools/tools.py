from api.agents.rag.rag import (
    process_context,
    retrieve_data,
    process_reviews_context,
    retrieve_reviews_data,
)
from api.core.config import DEFAULT_TOP_K
from api.server.models import RAGRequestExtraOptions
from api.utils.tracing import hide_sensitive_inputs
from langsmith import traceable


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="get_formatted_context",
    run_type="tool",
)
def get_formatted_context(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    extra_options = RAGRequestExtraOptions(top_k=top_k, enable_reranking=False)

    context = retrieve_data(query, extra_options)
    formatted_context = process_context(context)

    return formatted_context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="get_formatted_reviews_context",
    run_type="tool",
)
def get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    context = retrieve_reviews_data(query, item_list, top_k)
    formatted_context = process_reviews_context(context)

    return formatted_context
