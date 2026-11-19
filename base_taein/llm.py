"""Upstage API wrappers (chat + embeddings), based on RAG_Baseline.ipynb.

The Upstage API is OpenAI-compatible, so we use the `openai` client with
Upstage's base_url. The API key is read from the UPSTAGE_API_KEY env var.
"""
import os

from openai import OpenAI

# API key lookup order: UPSTAGE_API_KEY env var -> base_taein/.env file
# (one line: UPSTAGE_API_KEY=up_xxx). .env is git-ignored; never commit a key
# and leave it out of the submission ZIP (assignment rule).
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env_file(path=ENV_FILE):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "UPSTAGE_API_KEY":
                return value.strip().strip('"').strip("'") or None
    return None

BASE_URL = "https://api.upstage.ai/v1"
CHAT_MODEL = "solar-mini4-260922"          # required by the assignment
EMBED_PASSAGE_MODEL = "solar-embedding-2-passage"
EMBED_QUERY_MODEL = "solar-embedding-2-query"

MAX_TOKENS = 4096          # assignment upper bound is 4096
TEMPERATURE = 0
REASONING_EFFORT = "none"  # set to None to omit the parameter entirely

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("UPSTAGE_API_KEY") or _read_env_file()
        if not api_key:
            raise RuntimeError(
                "No Upstage API key. Create base_taein/.env with the line "
                "UPSTAGE_API_KEY=your-key, or in PowerShell: "
                '$env:UPSTAGE_API_KEY="your-key"'
            )
        _client = OpenAI(api_key=api_key, base_url=BASE_URL,
                         max_retries=3, timeout=120)
    return _client


def chat(messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """One LLM call. Returns the message text ('' if empty)."""
    assert max_tokens <= 4096, "assignment limit: max_tokens <= 4096"
    kwargs = dict(model=CHAT_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if REASONING_EFFORT is not None:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def embed(texts, model, batch_size=32):
    """Embed a list of strings. Returns a list of float vectors in input order."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = get_client().embeddings.create(
            model=model, input=batch, encoding_format="float")
        vectors.extend(item.embedding
                       for item in sorted(response.data, key=lambda d: d.index))
    return vectors
