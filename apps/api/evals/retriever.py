# Evals for the retriever component of the RAG pipeline.

from api.core.config import config
from api.core.constants import OPENAI, MODELS
from api.server.models import RAGRequest
from api.agents.rag.rag import rag_pipeline
import os

from langsmith import Client

from api.utils.metrics import (
    ragas_context_precision_id_based,
    ragas_context_recall_id_based,
    ragas_faithfulness,
    ragas_response_relevancy,
)

os.environ["LANGCHAIN_CONCURRENCY_LIMIT"] = "10"
# disable langsmith tracing for evals
os.environ["LANGSMITH_TRACING"] = "false"


ls_client = Client(api_key=config.LANGSMITH_API_KEY)


def target_function(inputs: dict) -> dict:
    try:
        payload = RAGRequest(
            query=inputs["question"],
            provider=OPENAI,
            model_name=MODELS[OPENAI][0],  # Use first available model from MODELS
            extra_options=None,  # use the default extra options
        )

        result = rag_pipeline(
            app_config=config,
            payload=payload,
        )
        return result
    except Exception as e:
        print(f"ERROR in target_function: {type(e).__name__}: {e}")
        raise


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
