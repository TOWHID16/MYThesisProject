# 00_gemini_rest_check.py
import os, requests, json, time
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
BASE = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 45  # seconds

def call_model(model_name: str, prompt: str) -> str:
    url = f"{BASE}/{model_name}:generateContent?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def main():
    assert API_KEY, "GEMINI_API_KEY missing in .env"
    models_try = ["models/gemini-2.5-pro", "models/gemini-2.5-flash"]
    prompt = "Return only this SQL: SELECT 1;"

    print("[i] Trying models (in order):", models_try)
    last_err = None
    for m in models_try:
        try:
            print(f"[i] Calling: {m}")
            txt = call_model(m, prompt)
            print(f"[OK] {m} ->", txt.strip())
            return
        except Exception as e:
            last_err = e
            print(f"[ERROR] {m} -> {repr(e)}")
            time.sleep(1)

    print("\n[FAIL] All candidates failed.")
    if last_err:
        print("Last error:", repr(last_err))

if __name__ == "__main__":
    main()
