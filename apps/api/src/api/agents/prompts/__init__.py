from typing import Protocol


# Define a protocol for prompt functions
class __PromptFunction(Protocol):
    def __call__(self, preprocessed_context: str, question: str) -> str: ...


SHOPPING_ASSISTANT_RAG_PROMPT_KEY = "SHOPPING_ASSISTANT_RAG_PROMPT"


def __shopping_assistant_rag_prompt(preprocessed_context: str, question: str) -> str:
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt


PROMPTS: dict[str, __PromptFunction] = {
    SHOPPING_ASSISTANT_RAG_PROMPT_KEY: __shopping_assistant_rag_prompt
}
