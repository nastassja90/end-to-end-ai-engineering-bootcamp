from typing import Optional
import numpy as np
import openai
from pydantic import BaseModel
from cohere import ClientV2 as CohereClient, V2RerankResponse as CohereRerankResponse

from api.core.config import Config
from api.server.models import RAGRequest, RAGUsedContextItem, RAGRequestExtraOptions
from api.utils.tracing import hide_sensitive_inputs
from api.utils.prompts import prompt_template_config
from api.utils.llm import (
    run_llm,
    extract_usage_metadata,
    StructuredResponse,
)

from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    Document,
    FusionQuery,
)

# use the qdrant collection that supports the hybrid search
collection_name = "Amazon-items-collection-01-hybrid-search"

prompt_yaml_filepath = "api/agents/prompts/rag_prompt_templates.yaml"
"""Path to the YAML file containing RAG prompt templates."""

# the text embedding model to use for both indexing and querying
embedding_model = "text-embedding-3-small"

# reranking model name
reranking_model = "rerank-v4.0-fast"

# instance for the cohere client
cohere_client: Optional[CohereClient] = None
"""Cohere client instance for re-ranking, initialized on first use."""


class RAGRetrievedContext(BaseModel):
    retrieved_context_ids: list[str] = list()
    retrieved_context: list[str] = list()
    similarity_scores: list[float] = list()
    retrieved_context_ratings: list[Optional[float]] = list()


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": embedding_model},
)
def get_embedding(text: str, model: str = embedding_model) -> list[float]:
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="rerank",
    run_type="retriever",
)
def rerank(
    query: str,
    original_context: RAGRetrievedContext,
    top_k: int,
) -> RAGRetrievedContext:
    """
    Rerank a set of retrieved contexts based on relevance to a query using a Cohere reranking model.

    Args:
        query (str): The input query string to rerank the contexts against.
        original_context (RAGRetrievedContext): The original context object containing retrieved documents, their IDs, similarity scores, and ratings.
        top_k (int): The number of top results to return after reranking.

    Returns:
        RAGRetrievedContext: A new context object containing the reranked documents, their corresponding IDs, similarity scores, and ratings, all ordered according to the reranking results.

    Notes:
        - This function uses a global `cohere_client` for reranking. If not initialized, it creates a new `CohereClient` instance.
        - The reranked context preserves the association between documents, their IDs, similarity scores, and ratings.
        - Consider refactoring to use dependency injection for better testability and maintainability.
    """
    reranked_context = RAGRetrievedContext()

    # initialize the cohere client if not already done
    # use global to modify the module-level cohere_client variable from the scope of this function
    # TODO: refactor this to use dependency injection.
    global cohere_client
    if not cohere_client:
        cohere_client = CohereClient()

    rerank: CohereRerankResponse = cohere_client.rerank(
        model=reranking_model,
        query=query,
        documents=original_context.retrieved_context,
        top_n=top_k,
    )

    # assign the reranked results to the retrieved_context list
    reranked_context.retrieved_context = [
        original_context.retrieved_context[result.index] for result in rerank.results
    ]

    # now based on the order of the reranked_context.retrieved_context, we must rearrange also
    # the retrieved_context_ids and similarity_scores lists from the original_context
    for element in reranked_context.retrieved_context:
        # find the index of this element in the original context
        original_index = original_context.retrieved_context.index(element)
        # use this index to get the corresponding id and similarity score
        reranked_context.retrieved_context_ids.append(
            original_context.retrieved_context_ids[original_index]
        )
        reranked_context.similarity_scores.append(
            original_context.similarity_scores[original_index]
        )
        # also the ratings
        reranked_context.retrieved_context_ratings.append(
            original_context.retrieved_context_ratings[original_index]
        )
        # now the reranked context is ordered with the same order as the reranking results

    return reranked_context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="retrieve_data",
    run_type="retriever",
)
def retrieve_data(
    query: str,
    qdrant_client: QdrantClient,
    extra_options: RAGRequestExtraOptions,
) -> RAGRetrievedContext:

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        # prefetch retrieves data using both vector and sparse search and it applies before
        # the actual fusion of the results happens (for costs optimization).
        # prefetches are multiple independent searches that get combined later
        prefetch=[
            # semantic search using vector embeddings
            Prefetch(query=query_embedding, using=embedding_model, limit=20),
            # sparse search using BM25
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20
            ),
        ],
        # RRF=Reciprocal Rank Fusion method to combine results based on their ranks in both searches
        # it applies after prefetching (20 vectors from each search in this case). This method also
        # orders the results based on their combined relevance, so that the first k results are the most relevant ones.
        query=FusionQuery(fusion="rrf"),
        # the final number of results to return after fusion are only the top k (5 in this case)
        limit=extra_options.top_k,
    )

    context: RAGRetrievedContext = RAGRetrievedContext()

    for result in results.points:
        context.retrieved_context_ids.append(result.payload["parent_asin"])
        context.retrieved_context.append(result.payload["description"])
        context.retrieved_context_ratings.append(result.payload["average_rating"])
        context.similarity_scores.append(result.score)

    # If enabled, apply re-ranking using Cohere models to the retrieved context
    if extra_options.enable_reranking is True:
        return rerank(query, context, extra_options.top_k)
    return context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="format_retrieved_context",
    run_type="prompt",
)
def process_context(context: RAGRetrievedContext) -> str:

    formatted_context = ""

    for id, chunk, rating in zip(
        context.retrieved_context_ids,
        context.retrieved_context,
        context.retrieved_context_ratings,
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="build_prompt",
    run_type="prompt",
)
def build_prompt(prompt_key: str, preprocessed_context: str, question: str) -> str:
    template = prompt_template_config(prompt_yaml_filepath, prompt_key)
    prompt = template.render(
        preprocessed_context=preprocessed_context, question=question
    )
    return prompt


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="generate_answer",
    run_type="llm",
)
def generate_answer(
    app_config: Config,
    provider: str,
    model_name: str,
    prompt: str,
) -> StructuredResponse:
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["ls_provider"] = provider
        current_run.metadata["ls_model_name"] = model_name

    output, original_response = run_llm(
        app_config,
        provider,
        model_name,
        messages=[{"role": "system", "content": prompt}],
    )

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            original_response, provider
        )

    return output


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="rag_pipeline",
)
def rag_pipeline(
    app_config: Config,
    payload: RAGRequest,
    qdrant_client: QdrantClient,
) -> dict:

    # retrieve the extra options from the payload; if not provided, use default values
    extra_options: RAGRequestExtraOptions = (
        payload.extra_options or RAGRequestExtraOptions()
    )

    retrieved_context: RAGRetrievedContext = retrieve_data(
        payload.query,
        qdrant_client,
        extra_options,
    )
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(
        app_config.RAG_PROMPT_KEY, preprocessed_context, payload.query
    )
    output: StructuredResponse = generate_answer(
        app_config, payload.provider, payload.model_name, prompt
    )

    result = {
        "data_model": output,
        "answer": output.answer,
        "references": output.references,
        "question": payload.query,
        "retrieved_context_ids": retrieved_context.retrieved_context_ids,
        "retrieved_context": retrieved_context.retrieved_context,
        "similarity_scores": retrieved_context.similarity_scores,
    }

    # enrich references with image_url and price from qdrant
    used_context: list[RAGUsedContextItem] = []

    for item in result.get("references", []):
        # find each reference item in qdrant to get its image_url and price, use the parent_asin field
        # as the matching value for the search
        # since we need to make an hybrid search query, we can't use scroll method, so we do a query_points call
        found = (
            qdrant_client.query_points(
                collection_name=collection_name,
                query=np.zeros(1536).tolist(),
                limit=1,
                using=embedding_model,
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

        # we expect only one point to be found
        image_url = found.get("image", None)
        price = found.get("price", 0.0)
        used_context.append(
            RAGUsedContextItem(
                image_url=image_url,
                price=price,
                description=item.description,
            )
        )

    return {
        "answer": result["answer"],
        "used_context": used_context,
    }
