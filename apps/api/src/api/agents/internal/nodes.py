from typing import Optional
from api.agents.internal.models import IntentRouterResponse, State
from langsmith import traceable, get_current_run_tree

from langchain_core.messages import convert_to_openai_messages
from openai import OpenAI
import instructor
from api.core.config import Config, OPENAI

from api.utils.tracing import hide_sensitive_inputs
from api.agents.prompts.prompts import prompt_template_config
from api.utils.utils import format_ai_message
from api.core.llm import (
    run_llm,
    extract_usage_metadata,
)

import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

qa_agent_prompt = "qa_agent"
"""Prompt ID containing the QA agent prompt template."""
intent_router_agent_prompt = "intent_router_agent"
"""Prompt ID containing the Intent Router agent prompt template."""

### QnA Agent Node


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="agent_node",
    run_type="llm",
    metadata={},
)
def agent_node(
    state: State, app_config: Config, provider: str, model_name: str
) -> dict:

    try:
        current_run = get_current_run_tree()
        if current_run:
            current_run.metadata["ls_provider"] = provider
            current_run.metadata["ls_model_name"] = model_name

        template = prompt_template_config(qa_agent_prompt)
        prompt = template.render(
            available_tools=state.available_tools, top_k=state.top_k
        )

        acc = []
        for message in state.messages:
            acc.append(convert_to_openai_messages(message))
        messages = [{"role": "system", "content": prompt}, *acc]

        logger.info(f"Agent Node - Messages: {messages}")

        logger.info(f"Invoking LLM with model: {model_name} from provider: {provider}")

        response, original_response = run_llm(
            app_config=app_config,
            provider=provider,
            model_name=model_name,
            messages=messages,
            temperature=0.5,
        )

        logger.info(f"LLM Response: {response}")

        if current_run:
            current_run.metadata["usage_metadata"] = extract_usage_metadata(
                original_response, provider
            )

        return {
            "messages": [format_ai_message(response)],
            "tool_calls": response.tool_calls,
            "iteration": state.iteration + 1,
            "answer": response.answer,
            "final_answer": response.final_answer,
            "references": response.references,
        }

    except Exception as exp:
        return {
            "messages": state.messages,
            "tool_calls": state.tool_calls,
            "iteration": state.iteration + 1,
            "answer": f"Error: {exp.args[0]}",
            "final_answer": True,
            "references": state.references,
        }


### Intent Router Agent Node


@traceable(
    process_inputs=hide_sensitive_inputs,
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": f"{OPENAI}", "ls_model_name": "gpt-4.1-mini"},
)
def intent_router_node(state: State):
    current_run = get_current_run_tree()

    template = prompt_template_config(intent_router_agent_prompt)

    prompt = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    # store here the current trace id from langsmith
    trace_id: Optional[str] = None

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            raw_response, OPENAI
        )
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id,
    }
