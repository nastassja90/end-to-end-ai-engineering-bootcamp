from typing import Optional
import numpy as np
import openai
from cohere import V2RerankResponse as CohereRerankResponse

from api.core.cohere import cohere_client
from api.agents.internal.models import RAGRetrievedContext
from api.core.config import (
    RAG_COLLECTIONS,
    RAG_EMBEDDING_MODEL,
    RAG_RERANKING_MODEL,
    Config,
    config,
)
from api.core.qdrant import qdrant_client
from api.server.models import RAGRequest, RAGUsedContextItem, RAGRequestExtraOptions
from api.utils.tracing import hide_sensitive_inputs
from api.agents.prompts.prompts import prompt_template_config
from api.core.llm import (
    run_llm,
    extract_usage_metadata,
    StructuredResponse,
)

from langsmith import traceable, get_current_run_tree
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    Document,
    FusionQuery,
    MatchAny,
)


retrieval_generation_prompt = "retrieval_generation"
"""Prompt ID containing RAG prompt templates."""


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": RAG_EMBEDDING_MODEL},
)
def get_embedding(text: str, model: str = RAG_EMBEDDING_MODEL) -> list[float]:
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
        - The reranked context preserves the association between documents, their IDs, similarity scores, and ratings.
        - Consider refactoring to use dependency injection for better testability and maintainability.
    """
    reranked_context = RAGRetrievedContext()

    rerank: CohereRerankResponse = cohere_client.get().rerank(
        model=RAG_RERANKING_MODEL,
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


@traceable(name="retrieve_reviews_data", run_type="retriever")
def retrieve_reviews_data(query, item_list, k=5):

    query_embedding = get_embedding(query)

    results = qdrant_client.get().query_points(
        collection_name=RAG_COLLECTIONS["reviews"],
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(key="parent_asin", match=MatchAny(any=item_list))
                    ]
                ),
                limit=20,
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["text"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


@traceable(name="format_retrieved_reviews_context", run_type="prompt")
def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(
        context["retrieved_context_ids"], context["retrieved_context"]
    ):
        formatted_context += f"- ID: {id}, review: {chunk}\n"

    return formatted_context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="retrieve_data",
    run_type="retriever",
)
def retrieve_data(
    query: str,
    extra_options: RAGRequestExtraOptions,
) -> RAGRetrievedContext:

    query_embedding = get_embedding(query)

    results = qdrant_client.get().query_points(
        collection_name=RAG_COLLECTIONS["items"],
        # prefetch retrieves data using both vector and sparse search and it applies before
        # the actual fusion of the results happens (for costs optimization).
        # prefetches are multiple independent searches that get combined later
        prefetch=[
            # semantic search using vector embeddings
            Prefetch(query=query_embedding, using=RAG_EMBEDDING_MODEL, limit=20),
            # sparse search using BM25
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20
            ),
        ],
        # RRF=Reciprocal Rank Fusion method to combine results based on their ranks in both searches
        # it applies after prefetching (20 vectors from each search in this case). This method also
        # orders the results based on their combined relevance, so that the first k results are the most relevant ones.
        query=FusionQuery(fusion="rrf"),
        # the final number of results to return after fusion are only the top k
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
def build_prompt(preprocessed_context: str, question: str) -> str:
    template = prompt_template_config(retrieval_generation_prompt)
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
) -> dict:

    current_run = get_current_run_tree()

    # retrieve the extra options from the payload; if not provided, use default values
    extra_options: RAGRequestExtraOptions = (
        payload.extra_options or RAGRequestExtraOptions()
    )

    retrieved_context: RAGRetrievedContext = retrieve_data(
        payload.query,
        extra_options,
    )
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, payload.query)
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
            qdrant_client.get()
            .query_points(
                collection_name=RAG_COLLECTIONS["items"],
                query=np.zeros(1536).tolist(),
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

    # store here the current trace id from langsmith
    trace_id: Optional[str] = None
    if current_run:
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    return {
        "answer": result["answer"],
        "used_context": used_context,
        "question": payload.query,
        "retrieved_context_ids": retrieved_context.retrieved_context_ids,
        "retrieved_context": retrieved_context.retrieved_context,
        "trace_id": trace_id,
    }
