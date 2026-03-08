from typing import Literal, Optional, cast

from langgraph.graph import END
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt
from api.agents.prompts.prompts import prompt_template_config
from api.agents.common.models import (
    StateAdvanced as State,
    ProductQAAgentResponse,
    ShoppingCartAgentResponse,
    CoordinatorAgentResponse,
    WarehouseManagerAgentResponse,
)
from api.utils.utils import format_ai_message
from api.core.llm import (
    run_llm,
    extract_usage_metadata,
)
from api.utils.logs import logger


product_qa_agent_prompt = "product_qa_agent"
"""Prompt ID containing the Product QA agent prompt template."""
shopping_cart_agent_prompt = "shopping_cart_agent"
"""Prompt ID containing the Shopping Cart agent prompt template."""
warehouse_manager_agent_prompt = "warehouse_manager_agent"
"""Prompt ID containing the Warehouse Manager agent prompt template."""
coordinator_agent_prompt = "coordinator_agent"
"""Prompt ID containing the Coordinator agent prompt template."""


@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={},
)
def product_qa_agent(state: State, provider: str, model_name: str) -> dict:

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["ls_provider"] = provider
        current_run.metadata["ls_model_name"] = model_name

    logger.info(f"Product QA Agent Node - State Messages: {state.messages}")

    logger.info(f"Invoking LLM with model: {model_name} from provider: {provider}")

    response, original_response = run_llm(
        provider=provider,
        model_name=model_name,
        messages=state.messages,
        prompts=prompt_template_config(product_qa_agent_prompt),
        prompt_template_parameters={
            "available_tools": state.product_qa_agent.available_tools,
            "top_k": state.top_k,
            "enable_reranking": state.enable_reranking,
        },
        temperature=0.5,
        response_model=ProductQAAgentResponse,
    )

    response = cast(ProductQAAgentResponse, response)

    logger.info(f"LLM Response: {response}")

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            original_response, provider
        )

    return {
        "messages": [format_ai_message(response)],
        "product_qa_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.product_qa_agent.available_tools,
        },
        "answer": response.answer,
        "references": response.references,
    }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={},
)
def shopping_cart_agent(state: State, provider: str, model_name: str) -> dict:

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["ls_provider"] = provider
        current_run.metadata["ls_model_name"] = model_name

    logger.info(f"Shopping Cart Agent Node - State Messages: {state.messages}")

    logger.info(f"Invoking LLM with model: {model_name} from provider: {provider}")

    response, original_response = run_llm(
        provider=provider,
        model_name=model_name,
        messages=state.messages,
        prompts=prompt_template_config(shopping_cart_agent_prompt),
        prompt_template_parameters={
            "available_tools": state.shopping_cart_agent.available_tools,
            "user_id": state.user_id,
            "cart_id": state.cart_id,
        },
        temperature=0.5,
        response_model=ShoppingCartAgentResponse,
    )

    response = cast(ShoppingCartAgentResponse, response)

    logger.info(f"LLM Response: {response}")

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            original_response, provider
        )

    return {
        "messages": [format_ai_message(response)],
        "shopping_cart_agent": {
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "available_tools": state.shopping_cart_agent.available_tools,
        },
        "answer": response.answer,
    }


#### HITL Nodes


@traceable(name="hitl_add_to_cart")
def hitl_add_to_cart(
    state: State,
) -> Command[Literal["shopping_cart_agent_tool_node", f"{END}"]]:

    for tool_call in state.shopping_cart_agent.tool_calls:
        if tool_call.name == "add_to_shopping_cart":
            items_to_add = tool_call.arguments.items
            break

    human_input: dict = interrupt({"items_to_add": items_to_add})

    if human_input.get("confirmed"):
        return Command(update={}, goto="shopping_cart_agent_tool_node")
    else:

        last_msg = state.messages[-1]
        sanitized = AIMessage(
            content=last_msg.content,
            id=last_msg.id,  # same id = replace instead of append
        )

        return Command(
            update={
                "messages": [sanitized],
                "answer": "You have rejected the addition of items to the cart.",
            },
            goto=END,
        )


@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={},
)
def warehouse_manager_agent(state: State, provider: str, model_name: str) -> dict:

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["ls_provider"] = provider
        current_run.metadata["ls_model_name"] = model_name

    logger.info(f"Warehouse Manager Agent Node - State Messages: {state.messages}")

    logger.info(f"Invoking LLM with model: {model_name} from provider: {provider}")

    response, original_response = run_llm(
        provider=provider,
        model_name=model_name,
        messages=state.messages,
        prompts=prompt_template_config(warehouse_manager_agent_prompt),
        prompt_template_parameters={
            "available_tools": state.warehouse_manager_agent.available_tools,
            "user_id": state.user_id,
            "cart_id": state.cart_id,
        },
        temperature=0.5,
        response_model=WarehouseManagerAgentResponse,
    )

    response = cast(WarehouseManagerAgentResponse, response)

    logger.info(f"LLM Response: {response}")

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            original_response, provider
        )

    return {
        "messages": [format_ai_message(response)],
        "warehouse_manager_agent": {
            "iteration": state.warehouse_manager_agent.iteration + 1,
            "final_answer": response.final_answer,
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "available_tools": state.warehouse_manager_agent.available_tools,
        },
        "answer": response.answer,
    }


@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={},
)
def coordinator_agent(state: State, provider: str, model_name: str) -> dict:

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["ls_provider"] = provider
        current_run.metadata["ls_model_name"] = model_name

    logger.info(f"Coordinator Agent Node - State Messages: {state.messages}")

    logger.info(f"Invoking LLM with model: {model_name} from provider: {provider}")

    response, original_response = run_llm(
        provider=provider,
        model_name=model_name,
        messages=state.messages,
        prompts=prompt_template_config(coordinator_agent_prompt),
        prompt_template_parameters={},
        temperature=0.5,
        response_model=CoordinatorAgentResponse,
    )

    response = cast(CoordinatorAgentResponse, response)

    logger.info(f"LLM Response: {response}")

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            original_response, provider
        )

    # store here the current trace id from langsmith
    trace_id: Optional[str] = None

    ai_message = []
    if response.final_answer:
        ai_message = [AIMessage(content=response.answer)]

    if current_run:
        current_run.metadata["usage_metadata"] = extract_usage_metadata(
            original_response, provider
        )
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    return {
        "messages": ai_message,
        "answer": response.answer,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": [data.model_dump() for data in response.plan],
        },
        "trace_id": trace_id,
    }
