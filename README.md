# End-to-End AI Engineering Bootcamp

A hands-on project featuring a **RAG (Retrieval-Augmented Generation) pipeline** with FastAPI, Streamlit, and Qdrant—complete with evaluation metrics via Ragas and observability through Langsmith.

---

## Quick Start

### 1. Clone and configure environment

```bash
cp env.example .env
```

Edit `.env` with your API keys:

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=your-project-name
```

### 2. Install dependencies

```bash
uv sync
source .venv/bin/activate
```

### 3. Authenticate with GitHub Container Registry

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

> **Note:** `GITHUB_TOKEN` should be a [Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with `read:packages` scope, exported in your shell profile (`.zshrc`, `.bashrc`, etc.).

### 4. Launch the stack

```bash
make run-docker-compose
```

Once running, access:

| Service          | URL                             |
| ---------------- | ------------------------------- |
| Streamlit UI     | http://localhost:8501           |
| FastAPI docs     | http://localhost:8000/docs      |
| Qdrant dashboard | http://localhost:6333/dashboard |

Now you are ready to [embed the first data into Qdrant](#preparing-the-dataset).

---

## Prerequisites

Before you begin, ensure you have:

- **Python** with [uv](https://github.com/astral-sh/uv) package manager
- **Docker** and **Docker Compose**
- API accounts for **OpenAI**, **Google AI (Gemini)**, and **Groq**
- A **Langsmith** account with a project created for this repository

---

## Project Architecture

The Docker Compose setup includes three services:

- **FastAPI** — Backend implementing the RAG pipeline
- **Streamlit** — Frontend chat interface for interacting with the LLM
- **Qdrant** — Vector database for storing embeddings (persisted in [qdrant_storage](./qdrant_storage))

---

## Dataset Setup

Before developing the RAG pipeline, you need to embed initial data into Qdrant.

### Preparing the dataset

Follow these notebooks in order:

1. **[01-explore-amazon-dataset.ipynb](./notebooks/week1/01-explore-amazon-dataset.ipynb)** — Explore and understand the raw Amazon dataset
2. **[02-RAG-preprocessing-amazon-dataset.ipynb](./notebooks/week1/02-RAG-preprocessing-amazon-dataset.ipynb)** — Preprocess and embed the data into Qdrant

### Generating synthetic evaluation data

To test your RAG pipeline, you can generate synthetic question-answer pairs using an LLM:

- **[04-RAG-evaluation-synthetic-dataset.ipynb](./notebooks/week1/04-RAG-evaluation-synthetic-dataset.ipynb)** — Create evaluation datasets programmatically

## Running Evaluations

Execute retriever evaluations with:

```bash
make run-evals-retriever
```

Evaluation metrics are computed using [Ragas](https://docs.ragas.io/en/stable/#why-ragas). Results and experiment history are available in your [Langsmith](https://smith.langchain.com/) dashboard under **Datasets & Experiments**.

Application traces for the entire RAG pipeline can be found under the **Tracing** section.

---

## Useful Links

- [Langsmith Dashboard](https://smith.langchain.com/)
- [Ragas Documentation](https://docs.ragas.io/en/stable/)
- [uv Package Manager](https://github.com/astral-sh/uv)
