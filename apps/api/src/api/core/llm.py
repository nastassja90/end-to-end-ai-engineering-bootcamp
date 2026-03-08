from typing import Type, TypeAlias, Tuple, List, Dict, Any
from jinja2 import Template
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from pydantic import BaseModel
from groq.types.chat import ChatCompletion as GroqChatCompletion
from google.genai.types import GenerateContentResponse
from api.core.config import GOOGLE, MODELS, config
from google.genai import Client as Gemini
from api.agents.common.models import StructuredResponse
from instructor import Instructor, Mode, from_genai, from_litellm
from litellm import completion
from api.utils.logs import logger
from langchain_core.messages import convert_to_openai_messages


# Define a type alias for LLM responses
LLMResponse: TypeAlias = (
    OpenAIChatCompletion | GroqChatCompletion | GenerateContentResponse
)


def __is_google_model(model_name: str) -> bool:
    """Check if the given model name belongs to the Google provider."""
    if GOOGLE not in MODELS:
        return False
    return model_name in MODELS[GOOGLE]


def __models(provider: str, model_name: str) -> List[str]:
    models: List[str] = [model_name]
    for item in MODELS.items():
        p, mm = item
        if p == provider:
            for m in mm:
                if m != model_name:
                    models.append(m)
        else:
            models.extend(mm)
    return models


def convert_messages_for_gemini(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages to Gemini-compatible format.

    Gemini uses 'user' and 'model' roles (not 'assistant').
    System messages are converted to user messages with clear context.
    Tool messages need special handling to preserve the conversation flow.

    Args:
        messages: List of OpenAI-format messages with roles like 'system', 'user', 'assistant', 'tool'

    Returns:
        List of Gemini-compatible messages with 'user' and 'model' roles
    """
    gemini_messages = []
    system_content = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            # Store system message to prepend to the first user message
            system_content = content

        elif role == "user":
            # Prepend system content to first user message if exists
            if system_content:
                content = f"{system_content}\n\n---\n\nUser query: {content}"
                system_content = None
            gemini_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            # Convert assistant to model role for Gemini
            # Include tool_calls info in the content if present
            assistant_content = content or ""
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                tool_calls_str = "\n\nTool calls made:\n"
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    tool_args = func.get("arguments", "{}")
                    tool_calls_str += f"- {tool_name}: {tool_args}\n"
                assistant_content += tool_calls_str

            if assistant_content:
                gemini_messages.append({"role": "model", "content": assistant_content})

        elif role == "tool":
            # Convert tool response to user message with clear context
            tool_name = msg.get("name", "unknown_tool")
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = (
                f"Tool result from '{tool_name}' (call_id: {tool_call_id}):\n{content}"
            )
            gemini_messages.append({"role": "user", "content": tool_content})

    # If only system message was provided (pipeline mode), convert it to a user message
    if system_content and not gemini_messages:
        gemini_messages.append({"role": "user", "content": system_content})

    # Gemini requires alternating user/model messages, merge consecutive same-role messages
    merged_messages = []
    for msg in gemini_messages:
        if merged_messages and merged_messages[-1]["role"] == msg["role"]:
            # Merge with previous message of same role
            merged_messages[-1]["content"] += f"\n\n{msg['content']}"
        else:
            merged_messages.append(msg)

    return merged_messages


def extract_usage_metadata(response: LLMResponse, provider: str) -> dict:
    """Extract usage metadata from different LLM provider responses."""
    try:
        logger.info(f"LLM Response: {response}")
        if provider == GOOGLE:
            return {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
                "cached_tokens": response.usage_metadata.cache_tokens_details,
            }
        else:  # Supports both GROQ and OPENAI
            return {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_tokens": response.usage.prompt_tokens_details.cached_tokens,
            }
    except Exception as e:
        logger.exception(f"Failed to extract usage metadata: {e}")
        return {}


def run_llm(
    provider: str,
    model_name: str,
    messages,
    prompts: Dict[str, str],
    prompt_template_parameters: Dict[str, Any] = {},
    temperature: float = 0.0,
    response_model: Type[BaseModel] = StructuredResponse,
) -> Tuple[BaseModel, LLMResponse]:
    """Run an LLM request using the configured provider and return a structured response.

    Selects the appropriate client for Google Gemini, Groq, or OpenAI, applies
    provider-specific modes and message conversions, and executes a chat completion
    request with a StructuredResponse schema.

    Args:
        provider: Provider identifier (e.g., GOOGLE, GROQ, or default OpenAI).
        model_name: Model name to use for the request.
        messages: Chat messages payload for the model.
        prompts: Dictionary of prompt templates for different agents.
        prompt_template_parameters: Dictionary of parameters to render the prompt templates.
        temperature: Sampling temperature for the request.
        response_model: Pydantic model class to parse the LLM response into a structured format.

    Returns:
        A tuple of (response_model instance, LLMResponse) from the completion call.

    Raises:
        Exception: Propagates any errors raised during the Gemini call after logging.
    """

    acc = []
    for message in messages:
        acc.append(convert_to_openai_messages(message))

    client: Instructor = from_litellm(completion)
    for model in __models(provider, model_name):

        try:
            ai_messages: List[dict] = []
            template = Template(prompts.get(model))
            prompt = template.render(**prompt_template_parameters)
            ai_messages = [{"role": "system", "content": prompt}, *acc]

            if __is_google_model(model):
                logger.info(
                    f"Using Google Gemini model '{model}' with GENAI_STRUCTURED_OUTPUTS mode"
                )
                # Use GENAI_STRUCTURED_OUTPUTS mode for Gemini to avoid MALFORMED_FUNCTION_CALL errors
                # Gemini gets confused when StructuredResponse contains tool_calls field with GENAI_TOOLS mode
                return from_genai(
                    Gemini(api_key=config.GOOGLE_API_KEY),
                    mode=Mode.GENAI_STRUCTURED_OUTPUTS,
                ).chat.completions.create_with_completion(
                    model=model_name,
                    messages=convert_messages_for_gemini(ai_messages),
                    response_model=response_model,
                )
            return client.chat.completions.create_with_completion(
                model=model,
                response_model=response_model,
                messages=ai_messages,
                temperature=temperature,
            )
        except Exception as e:
            logger.exception(f"Error with model {model}: {e}")
            continue
    raise Exception("All models failed")
