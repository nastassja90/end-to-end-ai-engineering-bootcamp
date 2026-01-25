# Evals for the retriever component of the RAG pipeline.

from api.core.config import config
from api.core.constants import OPENAI, MODELS
from api.server.models import RAGRequest, RAGRequestExtraOptions
from api.agents.retrieval_generation import rag_pipeline
import os

from qdrant_client import QdrantClient

from langsmith import Client
from qdrant_client import QdrantClient

from api.utils.ragas import (
    ragas_context_precision_id_based,
    ragas_context_recall_id_based,
    ragas_faithfulness,
    ragas_response_relevancy,
)

os.environ["LANGCHAIN_CONCURRENCY_LIMIT"] = "10"
# disable langsmith tracing for evals
os.environ["LANGSMITH_TRACING"] = "false"


ls_client = Client(api_key=config.LANGSMITH_API_KEY)
qdrant_client = QdrantClient(url=config.QDRANT_URL)


def target_function(inputs: dict) -> dict:
    payload = RAGRequest(
        query=inputs["question"],
        provider=OPENAI,
        model_name="gpt-5-nano",  # TODO: replace with models from MODELS[OPENAI]
        extra_options=None,  # use the default extra options
    )

    return rag_pipeline(
        app_config=config,
        payload=payload,
        qdrant_client=qdrant_client,
    )


results = ls_client.evaluate(
    target_function,
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
    ],
    experiment_prefix="retriever",
)
