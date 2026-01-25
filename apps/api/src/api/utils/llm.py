from typing import TypeAlias, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from groq import Groq
from groq.types.chat import ChatCompletion as GroqChatCompletion
from google.genai import Client as Gemini
from google.genai.types import GenerateContentResponse
from api.core.config import Config
from api.core.constants import GROQ, GOOGLE
from pydantic import BaseModel, Field
import instructor


#############################################################
# Define the Structured Output response type for Instructor #
#############################################################


# Define the pydantic model for the ReferencedItem retrieved from the vector database.
class ReferencedItem(BaseModel):
    id: str = Field(
        ..., description="The unique identifier of the referenced item (parent ASIN)."
    )
    description: str = Field(
        ..., description="The short description of the referenced item."
    )


# Define the output schema using Pydantic. This schema will be used to structure the model's response via instructor.
class StructuredResponse(BaseModel):
    answer: str = Field(
        ..., description="A brief summary of the weather in Italy today."
    )
    references: list[ReferencedItem] = Field(
        ..., description="A list of items used to answer the question."
    )


#############################################################


# Define a type alias for LLM responses
LLMResponse: TypeAlias = (
    OpenAIChatCompletion | GroqChatCompletion | GenerateContentResponse
)


def extract_usage_metadata(response: LLMResponse, provider: str) -> dict:
    """Extract usage metadata from different LLM provider responses."""
    if provider == GOOGLE:
        return {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }
    elif provider == GROQ:
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    else:  # Default to OpenAI
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }


def run_llm(
    app_config: Config,
    provider: str,
    model_name: str,
    messages,
    temperature: int = 0,
) -> Tuple[StructuredResponse, LLMResponse]:

    client: instructor.Instructor | None = None

    if provider == GOOGLE:
        client = instructor.from_genai(Gemini(api_key=app_config.GOOGLE_API_KEY))
        # For gemini we need a different format for messages and temperature is not supported
        return client.chat.completions.create_with_completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [message["content"] for message in messages],
                }
            ],
            response_model=StructuredResponse,
        )
    elif provider == GROQ:
        client = instructor.from_groq(Groq(api_key=app_config.GROQ_API_KEY))
    else:  # default to OpenAI
        client = instructor.from_openai(OpenAI(api_key=app_config.OPENAI_API_KEY))

    return client.chat.completions.create_with_completion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_model=StructuredResponse,
    )
