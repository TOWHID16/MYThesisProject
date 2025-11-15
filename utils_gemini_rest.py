# utils_gemini_rest.py
import os, time, random, hashlib, threading
from typing import List, Optional, Tuple, Union
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE = os.getenv("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")

# ENV override possible (comma-separated):
#   GEMINI_MODEL_ORDER="models/gemini-2.5-pro,models/gemini-2.5-flash,models/gemini-2.5-flash-lite"
_env_order = os.getenv("GEMINI_MODEL_ORDER")
DEFAULT_TRY_ORDER: List[str] = (
    [m.strip() for m in _env_order.split(",")] if _env_order
    else ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.5-flash-lite"]
)

# Debug toggle
DEBUG = os.getenv("GEMINI_DEBUG", "0") == "1"

# Reuse HTTP connection for efficiency
_session_lock = threading.Lock()
_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                s = requests.Session()
                # A tiny bump for robustness
                adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _session = s
    return _session

def _post_json(url: str, payload: dict, timeout: int) -> dict:
    sess = _get_session()
    r = sess.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _extract_text(resp_json: dict) -> str:
    # robust extraction
    try:
        cands = resp_json.get("candidates") or []
        if not cands:
            return str(resp_json)
        parts = cands[0].get("content", {}).get("parts") or []
        if parts and "text" in parts[0]:
            return parts[0]["text"]
        # fallback: join all text parts if exist
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        if any(texts):
            return "\n".join(texts)
        return str(resp_json)
    except Exception:
        return str(resp_json)

# ----------------------------------------------------------------------
# Lightweight in-memory cache (per-process)
# Key depends on (prompt, temperature, max_tokens, try_order tuple)
# ----------------------------------------------------------------------
_CACHE_MAX = int(os.getenv("GEMINI_CACHE_MAX", "256"))
_cache: dict[str, Tuple[str, str]] = {}   # key -> (text, model_used)
_cache_order: List[str] = []              # simple FIFO/LRU-ish

def _cache_key(prompt: str, temperature: float, max_output_tokens: int, try_order: List[str]) -> str:
    h = hashlib.sha256()
    # Avoid storing raw prompt if you care about memory/privacy; hash it
    h.update(prompt.encode("utf-8", errors="ignore"))
    h.update(str(float(temperature)).encode("ascii"))
    h.update(str(int(max_output_tokens)).encode("ascii"))
    h.update("|".join(try_order).encode("utf-8"))
    return h.hexdigest()

def _cache_get(key: str) -> Optional[Tuple[str, str]]:
    return _cache.get(key)

def _cache_put(key: str, value: Tuple[str, str]) -> None:
    if key in _cache:
        return
    _cache[key] = value
    _cache_order.append(key)
    # Trim
    if len(_cache_order) > _CACHE_MAX:
        old = _cache_order.pop(0)
        _cache.pop(old, None)

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def gemini_generate_text(
    prompt: str,
    models: Optional[List[str]] = None,
    timeout: int = 60,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    max_retries: int = 5,
    sleep_between_models: float = 0.4,
    return_model: bool = False,
) -> Union[str, Tuple[str, str]]:
    """
    Generic Gemini REST caller with retry/backoff, session reuse, cache, and model fallback.

    Args:
        prompt: input text.
        models: optional model try-order; defaults to DEFAULT_TRY_ORDER (can override via ENV).
        timeout: per-request timeout (seconds).
        temperature: sampling temperature.
        max_output_tokens: output token limit requested to the API.
        max_retries: attempts per model before moving to next.
        sleep_between_models: delay between switching models.
        return_model: if True, return (text, model_name). Otherwise just text.

    Returns:
        str or (str, model_name) depending on return_model.
    """
    assert GEMINI_API_KEY, "GEMINI_API_KEY missing in .env"
    try_order = models or DEFAULT_TRY_ORDER
    last_err: Optional[Exception] = None

    # Cache check
    ck = _cache_key(prompt, temperature, max_output_tokens, try_order)
    cached = _cache_get(ck)
    if cached is not None:
        text, used_model = cached
        if DEBUG:
            print(f"[gemini:cache] hit -> {used_model}")
        return (text, used_model) if return_model else text

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    for m in try_order:
        url = f"{GEMINI_BASE}/{m}:generateContent?key={GEMINI_API_KEY}"
        for attempt in range(max_retries):
            if DEBUG:
                print(f"[gemini] calling: {m} (attempt {attempt+1}/{max_retries})")
            try:
                data = _post_json(url, payload, timeout=timeout)
                text = _extract_text(data)
                _cache_put(ck, (text, m))
                return (text, m) if return_model else text
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                last_err = e
                # 429/503: soft failures → retry with backoff
                if status in (429, 503) and attempt < max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0, 0.35)
                    time.sleep(backoff)
                    continue
                # Other HTTP errors → try next model
                break
            except Exception as e:
                last_err = e
                # brief wait, then break to next model
                time.sleep(0.3)
                break
        # move to next model with a small gap
        time.sleep(sleep_between_models)

    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")
