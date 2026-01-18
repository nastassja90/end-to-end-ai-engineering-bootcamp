import openai
from api.core.config import Config
from api.server.models import RAGRequest
from api.agents.prompts import PROMPTS
from api.utils.llm import run_llm, extract_response_text, extract_usage_metadata
from api.utils.tracing import hide_sensitive_inputs
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
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
    name="retrieve_data",
    run_type="retriever",
)
def retrieve_data(query: str, qdrant_client: QdrantClient, k: int = 5) -> dict:

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="format_retrieved_context",
    run_type="prompt",
)
def process_context(context: dict) -> str:

    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="build_prompt",
    run_type="prompt",
)
def build_prompt(prompt_key: str, preprocessed_context: str, question: str) -> str:
    prompt = PROMPTS[prompt_key](preprocessed_context, question)
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
) -> str:
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["ls_provider"] = provider
        current_run.metadata["ls_model_name"] = model_name

    response = run_llm(
        app_config,
        provider,
        model_name,
        messages=[{"role": "system", "content": prompt}],
    )

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            response, provider
        )

    return extract_response_text(response, provider)


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="rag_pipeline",
)
def rag_pipeline(
    app_config: Config,
    payload: RAGRequest,
    qdrant_client: QdrantClient,
    top_k: int = 5,
) -> dict:

    retrieved_context = retrieve_data(payload.query, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(
        app_config.RAG_PROMPT_KEY, preprocessed_context, payload.query
    )
    answer = generate_answer(app_config, payload.provider, payload.model_name, prompt)

    return {
        "answer": answer,
        "question": payload.query,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"],
    }
