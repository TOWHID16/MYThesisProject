# 00_smoke_test_llms.py
# Smoke-test OpenAI & Gemini (Google AI Studio) with multiple models.

import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI, __version__ as openai_ver
import google.generativeai as genai
import google.generativeai as google_genai_pkg  # for __version__

print("[i] OPENAI_API_KEY set? ", bool(os.getenv("OPENAI_API_KEY")))
print("[i] GEMINI_API_KEY set? ", bool(os.getenv("GEMINI_API_KEY")))
print("[i] openai sdk version:", openai_ver)
print("[i] google-generativeai version:", google_genai_pkg.__version__)

# -------------------------------
# OpenAI: try multiple chat models
# -------------------------------
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-3.5-turbo",
]

print("\n==== OpenAI smoke ====")
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for m in OPENAI_MODELS:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Return only this SQL: SELECT 1;"}],
                temperature=0.0,
            )
            print(f"[OpenAI:{m}] OK ->", resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[OpenAI:{m}] ERROR:", repr(e))
            # Common: RateLimitError/insufficient_quota if billing not set
except Exception as e:
    print("[OpenAI] FATAL:", repr(e))

# -----------------------------------------
# Gemini (Google AI Studio): list & try one
# -----------------------------------------
print("\n==== Gemini smoke ====")
try:
    genai.configure(
        api_key=os.getenv("GEMINI_API_KEY"),
        client_options={"api_endpoint": "https://generativelanguage.googleapis.com"},
    )

    # List available models that support generateContent
    available = []
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)
    except Exception as e:
        print("[Gemini] list_models ERROR:", repr(e))

    print("[Gemini] Available models:", available if available else "(none)")

    # Prefer 1.5 family; else fall back to older names if present
    long_names = [
        "models/gemini-1.5-pro",
        "models/gemini-1.5-pro-001",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-001",
        "models/gemini-1.0-pro",
        "models/gemini-pro",
    ]
    short_aliases = [
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    preferred = None
    # Try full resource names first
    for cand in long_names:
        if cand in available:
            preferred = cand
            break
    # If list_models returns short names, try those
    if preferred is None:
        for cand in short_aliases:
            if cand in available:
                preferred = cand
                break
    # Fallback to a commonly available alias
    if preferred is None:
        preferred = "gemini-1.5-flash"

    print(f"[Gemini] Trying model: {preferred}")
    model = genai.GenerativeModel(preferred)
    resp = model.generate_content("Return only this SQL: SELECT 1;")
    print("[Gemini] OK ->", (resp.text or "").strip())

except Exception as e:
    print("[Gemini] ERROR:", repr(e))
    print("    Hint: Use an AI Studio key (aistudio.google.com). If you still see v1beta/NotFound, "
          "upgrade google-generativeai and re-create the key.")
