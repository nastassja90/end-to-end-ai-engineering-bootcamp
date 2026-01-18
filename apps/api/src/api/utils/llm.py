from typing import TypeAlias
from openai import OpenAI
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from groq import Groq
from groq.types.chat import ChatCompletion as GroqChatCompletion
from google.genai import Client as Gemini
from google.genai.types import GenerateContentResponse
from api.core.config import Config
from api.core.constants import GROQ, GOOGLE

# Define a type alias for the LLM client
LLMClient: TypeAlias = OpenAI | Groq | Gemini | None

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


def extract_response_text(response: LLMResponse, provider: str) -> str:
    """Extract the response text from different LLM provider responses."""
    if provider == GOOGLE:
        return response.text
    elif provider == GROQ:
        return response.choices[0].message.content
    else:  # Default to OpenAI
        return response.choices[0].message.content


def run_llm(
    app_config: Config, provider, model_name, messages, max_tokens=500
) -> LLMResponse:

    client: LLMClient = None

    if provider == GOOGLE:
        client = Gemini(api_key=app_config.GOOGLE_API_KEY)
        response: LLMResponse = client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages],
        )
        return response

    elif provider == GROQ:
        client = Groq(api_key=app_config.GROQ_API_KEY)
        response: LLMResponse = client.chat.completions.create(
            model=model_name, messages=messages, max_completion_tokens=max_tokens
        )
        return response

    else:
        client = OpenAI(api_key=app_config.OPENAI_API_KEY)
        response: LLMResponse = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal",
        )
        return response
