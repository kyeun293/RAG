"""Upstage API wrappers: chat (with a per-question call budget), embeddings,
and Document Parse (layout parser used instead of MinerU).

API key lookup: UPSTAGE_API_KEY env var -> taein_BookRAG/.env -> base_taein/.env
(.env files are git-ignored; never commit a key).
"""
import json
import os
import random
import re
import threading
import time

import httpx
from openai import OpenAI, RateLimitError

import config

_client = None
_lock = threading.Lock()


def _read_env_file(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "UPSTAGE_API_KEY":
                return value.strip().strip('"').strip("'") or None
    return None


def api_key():
    key = (os.environ.get("UPSTAGE_API_KEY")
           or _read_env_file(os.path.join(config.HERE, ".env"))
           or _read_env_file(os.path.join(config.HERE, "..", ".env")))
    if not key:
        raise RuntimeError("No Upstage API key: set UPSTAGE_API_KEY or create .env")
    return key


def get_client():
    global _client
    with _lock:
        if _client is None:
            _client = OpenAI(api_key=api_key(), base_url="https://api.upstage.ai/v1",
                             max_retries=3, timeout=180)
    return _client


class CallBudget:
    """Counts LLM calls for one question and refuses to exceed the limit."""

    def __init__(self, limit=config.MAX_LLM_CALLS_PER_QUESTION):
        self.limit, self.used = limit, 0

    def take(self):
        if self.used >= self.limit:
            raise RuntimeError(f"LLM call budget exceeded ({self.limit})")
        self.used += 1

    @property
    def left(self):
        return self.limit - self.used


def chat(messages, budget=None, max_tokens=config.MAX_TOKENS, json_mode=False):
    """One Solar Mini 4 call. `budget` is required at query time (None = offline
    indexing, which is not part of the per-question limit)."""
    assert max_tokens <= 4096, "assignment limit: max_tokens <= 4096"
    if budget is not None:
        budget.take()
    kwargs = dict(model=config.CHAT_MODEL, messages=messages,
                  temperature=config.TEMPERATURE, max_tokens=max_tokens)
    if config.REASONING_EFFORT is not None:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # Rate limits (429) are retried here with a longer backoff; a retried request
    # is the same single LLM invocation, so it is counted once in the budget.
    for attempt in range(6):
        try:
            response = get_client().chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == 5:
                raise
            time.sleep(min(60, 5 * 2 ** attempt) + random.random() * 3)


def parse_json(text):
    """Lenient JSON extraction from model output (handles ```json fences)."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def embed(texts, model, batch_size=32):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:12000] or " " for t in texts[start:start + batch_size]]
        response = get_client().embeddings.create(
            model=model, input=batch, encoding_format="float")
        vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    return vectors


def document_parse(pdf_bytes, filename="doc.pdf"):
    """Upstage Document Parse (sync, <=100 pages). Returns the JSON response."""
    response = httpx.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers={"Authorization": f"Bearer {api_key()}"},
        files={"document": (filename, pdf_bytes, "application/pdf")},
        data={"model": config.DOC_PARSE_MODEL, "mode": config.DOC_PARSE_MODE, "ocr": "auto",
              "output_formats": "['markdown', 'text']", "coordinates": "false"},
        timeout=600)
    response.raise_for_status()
    return response.json()
