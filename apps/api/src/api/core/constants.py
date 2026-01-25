OPENAI = "OpenAI"
GROQ = "Groq"
GOOGLE = "Google"

MODELS: dict[str, list[str]] = {
    OPENAI: ["gpt-4.1-mini"],
    GROQ: ["llama-3.3-70b-versatile"],
    GOOGLE: ["gemini-2.5-flash"],
}
