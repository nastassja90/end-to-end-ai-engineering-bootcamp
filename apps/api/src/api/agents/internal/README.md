# LangGraph Workflow - Shopping Assistant Agent

This documentation describes the implementation of the agentic workflow based on **LangGraph**, explaining how the various components interact and the rationale behind the architectural choices.

---

## Overview

The workflow implements a **Shopping Assistant** that answers questions about products in inventory. Unlike the classic RAG pipeline (which always executes retrieval → generation), the agentic approach allows the LLM to **autonomously decide** when and how to use the available tools.

```
┌─────────┐     ┌─────────────────────┐     ┌────────────┐     ┌─────────┐
│  START  │────▶│  intent_router_node │────▶│ agent_node │◀───▶│tool_node│
└─────────┘     └─────────────────────┘     └────────────┘     └─────────┘
                         │                         │
                         ▼                         ▼
                   (irrelevant                  (final
                     query)                    response)
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                                   ┌─────┐
                                   │ END │
                                   └─────┘
```

---

## Why LangGraph?

LangGraph was chosen for these reasons:

1. **Flow control**: Allows defining exactly when the agent should stop, retry, or pass to tools
2. **Shared state**: The `State` is accessible from all nodes, facilitating information passing
3. **Prebuilt components**: `ToolNode` automatically handles tool execution
4. **Debugging**: The graph is inspectable and traceable with Langsmith

---

## Workflow Nodes

### 1. Intent Router Node (`intent_router_node`)

**Purpose**: Filter irrelevant queries **before** activating the main agent.

**How it works**:

- Receives the user's query
- Uses a lightweight LLM (GPT-4.1-mini) to classify if the question is about products in inventory
- Returns `question_relevant: True/False`

**Why it exists**:

- Avoids wasting tokens and time on out-of-scope questions ("What's the capital of France?")
- Provides an immediate and polite response if the query is not relevant
- Separates routing logic from response logic

**Output**:

```python
{
    "question_relevant": bool,
    "answer": str  # Default response if not relevant
}
```

---

### 2. Agent Node (`agent_node`)

**Purpose**: Heart of the system — decides which tools to call and generates the final response.

**How it works**:

1. Receives the system prompt (from `qa_agent.yaml`) with the list of available tools
2. Analyzes the conversation (previous messages + tool results)
3. Decides whether to:
   - Call one or more tools (`tool_calls` populated)
   - Provide the final answer (`final_answer: True`)

**Why it's separate from the router**:

- The router makes a simple binary decision (relevant/not relevant)
- The agent performs complex reasoning (which tools to use, how to interpret results)
- Allows using different models for different tasks (cheap router, powerful agent)

**Output** (via Instructor/Pydantic):

```python
{
    "answer": str,
    "references": List[ReferencedItem],
    "tool_calls": List[ToolCall],
    "final_answer": bool
}
```

---

### 3. Tool Node (`tool_node`)

**Purpose**: Execute the tools requested by the agent.

**How it works**:

- Uses LangGraph's `ToolNode` (prebuilt component)
- Receives the list of `tool_calls` from state
- Executes each tool with the specified arguments
- Adds results to the `messages` list as `tool` type messages

**Why use prebuilt ToolNode**:

- Automatically handles tool message format
- Supports parallel execution of multiple tools
- Handles errors and formats them as agent-readable messages

---

## Workflow Edges

### Fixed Edges

| From        | To                   | Description                               |
| ----------- | -------------------- | ----------------------------------------- |
| `START`     | `intent_router_node` | Every request passes through filter first |
| `tool_node` | `agent_node`         | After tool execution, return to agent     |

### Conditional Edges

#### 1. After Intent Router

```python
def intent_router_conditional_edges(state):
    if state.question_relevant:
        return "agent_node"  # Proceed with agent
    else:
        return "end"  # Terminate with default response
```

**Scenarios**:

- ✅ "Do you sell earphones?" → `agent_node`
- ❌ "What's the weather today?" → `END` (response: "I can only help with products...")

#### 2. After Agent Node

```python
def tool_router(state):
    if state.final_answer:
        return "end"  # Agent has the answer
    elif state.iteration > 2:
        return "end"  # Anti-loop limit reached
    elif len(state.tool_calls) > 0:
        return "tools"  # Execute requested tools
    else:
        return "end"  # No tools, terminate
```

**Scenarios**:

- Agent calls tool → `tool_node` → returns to agent with results
- Agent has final answer → `END`
- Too many iterations (>2) → `END` (safety limit)

---

## The Tool: get_formatted_context

### What It Is

A **wrapper** around the original RAG pipeline, exposed as a callable tool for the agent.

### Why Wrap the RAG Pipeline?

The classic RAG pipeline (`retrieval_generation.py`) always executes:

```
Query → Embed → Search → Rerank → Format → Generate
```

By wrapping it in a tool, the agent can:

1. **Decide whether to use it**: Not all queries require retrieval
2. **Call it multiple times**: With different queries for complex questions
3. **Combine results**: From multiple calls for complete answers

### Implementation

```python
def get_formatted_context(query: str, top_k: int = 5) -> str:
    """
    Retrieves the top-k relevant products from the vector database.

    The agent automatically extracts search terms from the
    user's question (e.g., "Do you sell earphones?" → "earphones")
    """
    context = retrieve_data(query, qdrant_client, extra_options)
    formatted_context = process_context(context)
    return formatted_context
```

### Implicit Query Rewriting

The agent automatically performs **query rewriting**:

- User input: "Do you sell earphones?"
- Query to tool: "earphones"

This happens because the LLM understands that searching for products requires a semantic query, not a conversational question.

---

## The State

The state is the shared "container" across all nodes:

```python
class State(BaseModel):
    messages: List[Any]           # Conversation history
    question_relevant: bool       # Router output
    iteration: int                # Anti-loop counter
    answer: str                   # Current response
    available_tools: List[Dict]   # Tool descriptions for prompt
    tool_calls: List[ToolCall]    # Tools to execute
    final_answer: bool            # Termination flag
    references: List[ReferencedItem]  # Referenced products
```

**Note on `messages`**: Uses `Annotated[List, add]` to accumulate messages instead of overwriting them.

---

## Complete Flow: Example

**Query**: "Do you sell earphones?"

```
1. START
   └─▶ intent_router_node
       ├─ LLM classifies: "Question about products" ✅
       └─ Output: question_relevant=True

2. agent_node (iteration 1)
   ├─ LLM sees: tool available, no results yet
   ├─ Decides: call get_formatted_context(query="earphones")
   └─ Output: tool_calls=[ToolCall(name="get_formatted_context", arguments={query: "earphones"})]

3. tool_node
   ├─ Executes: get_formatted_context("earphones")
   ├─ Qdrant hybrid search + format
   └─ Output: 5 formatted products added to messages

4. agent_node (iteration 2)
   ├─ LLM sees: tool results with 5 products
   ├─ Decides: I have enough info to respond
   ├─ Generates detailed response with bullet points
   └─ Output: final_answer=True, answer="Yes! Here are our earphones..."

5. END
   └─ Final response returned to user
```

---

## Configuration and Customization

### Prompts

Prompts are in separate YAML files (`prompts/`):

- `qa_agent.yaml`: Instructions for the main agent
- `intent_router_agent.yaml`: Instructions for the filter

### Safety Limits

- **Max iterations**: 2 (prevents infinite loops if agent keeps calling tools)
- **Single tool**: Currently only `get_formatted_context`, but extensible

### Multi-Provider

The workflow works with OpenAI, Groq, and Gemini thanks to the abstraction in `run_llm()`.

**Gemini Note**: Requires specific message conversion (`convert_messages_for_gemini`) and explicit Pydantic models for tool arguments.

---

## Folder Files

| File        | Responsibility                                          |
| ----------- | ------------------------------------------------------- |
| `graph.py`  | LangGraph graph definition and edge conditions          |
| `nodes.py`  | Node implementations (agent_node, intent_router_node)   |
| `models.py` | Pydantic models for State, ToolCall, StructuredResponse |

---

## Future Extensions

1. **New tools**: Add tools for price filters, comparison, cart
2. **Memory**: Handle multi-turn conversations with persistent history
3. **Parallel tools**: Call multiple tools simultaneously for complex queries
4. **Human-in-the-loop**: Add nodes for human confirmation on critical actions
