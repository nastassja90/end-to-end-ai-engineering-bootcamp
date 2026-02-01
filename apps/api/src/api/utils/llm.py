from typing import TypeAlias, Tuple, List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from groq import Groq
from groq.types.chat import ChatCompletion as GroqChatCompletion
from google.genai import Client as Gemini
from google.genai.types import GenerateContentResponse
from api.core.config import Config
from api.core.constants import GROQ, GOOGLE
from api.agents.internal.models import StructuredResponse
import instructor


# Define a type alias for LLM responses
LLMResponse: TypeAlias = (
    OpenAIChatCompletion | GroqChatCompletion | GenerateContentResponse
)


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
    temperature: float = 0.0,
) -> Tuple[StructuredResponse, LLMResponse]:

    client: instructor.Instructor | None = None

    if provider == GOOGLE:
        client = instructor.from_genai(Gemini(api_key=app_config.GOOGLE_API_KEY))
        # Gemini requires specific message format conversion (roles, system handling, etc.)
        gemini_messages = convert_messages_for_gemini(messages)
        return client.chat.completions.create_with_completion(
            model=model_name,
            messages=gemini_messages,
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
