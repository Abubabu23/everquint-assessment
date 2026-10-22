import hashlib
import json
import os
import time

MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS",
    "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
).split(",")]
MAX_TRIES = 4
RETRY_CODES = (429, 500, 503)
CACHE_FILE = ".llm_cache.json"
USE_CACHE = os.getenv("LLM_CACHE", "1") == "1"

_client = None
_exhausted = set()
_cache = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def call_llm(prompt: str) -> str:
    from google.genai import errors

    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache = _load_cache()
    if USE_CACHE and key in cache:
        return cache[key]

    last_error = None
    for model in MODELS:
        if model in _exhausted:
            continue
        for attempt in range(MAX_TRIES):
            try:
                response = _get_client().models.generate_content(model=model, contents=prompt)
                text = response.text or ""
                if USE_CACHE and text:
                    cache[key] = text
                    _save_cache(cache)
                return text
            except errors.APIError as e:
                last_error = e
                if e.code == 404 or (e.code == 429 and "PerDay" in str(e)):
                    _exhausted.add(model)  # daily quota over / model not available
                    break
                if e.code not in RETRY_CODES:
                    raise
                time.sleep(2 ** (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("All Gemini models are out of free daily quota. Try again later.")