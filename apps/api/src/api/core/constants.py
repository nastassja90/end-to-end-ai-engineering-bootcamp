OPENAI = "OpenAI"
GROQ = "Groq"
GOOGLE = "Google"

MODELS: dict[str, list[str]] = {
    OPENAI: ["gpt-4.1-mini"],
    GROQ: ["llama-3.3-70b-versatile"],
    GOOGLE: ["gemini-2.5-flash"],
}

DEFAULT_TOP_K = 5
"""Default value for the number of relevant documents to retrieve in the RAG pipeline."""
MAX_TOP_K = 20
"""Maximum value for the number of relevant documents to retrieve in the RAG pipeline."""
