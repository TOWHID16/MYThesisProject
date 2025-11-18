# 2_run_baseline_cot.py
# Run a simple CoT baseline over runs/dev_600/my_test_set.json
# Outputs: runs/dev_600/predicted_cot_{modelname}.sql

import os, json, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils_gemini_rest import gemini_generate_text

# ---- CONFIG ----
PROJ_ROOT = Path(__file__).resolve().parent
RUN_DIR    = PROJ_ROOT / "runs" / "dev_600"
INPUT_TEST_SET = RUN_DIR / "my_test_set.json"
OUTPUT_FILE_PREFIX = "predicted_cot"

# Labeling-এর জন্য রাখলাম; আসল কলটা utils_gemini_rest ফ্যালব্যাক দিয়ে মডেল বেছে নেবে
MODELS_TO_TEST = [
    "models/gemini-2.5-pro",
]

TEMPERATURE_PLAN = 0.0   # single-pass CoT → 0.0
SLEEP_EVERY = 10         # প্রতি ১০ কুয়েরিতে ছোট বিরতি

def create_cot_prompt(question, schema_str):
    """
    Very simple CoT prompt (logical SQL order).
    """
    p = []
    p.append(schema_str.strip())
    p.append(f"### {question}")
    p.append("A: # Let's think step by step")
    p.append("# 1) Choose tables (FROM/JOIN).")
    p.append("# 2) Filter rows (WHERE).")
    p.append("# 3) Group if needed (GROUP BY).")
    p.append("# 4) Filter groups (HAVING).")
    p.append("# 5) Final columns (SELECT).")
    p.append("# Thus, the answer is:")
    p.append("```sql\nSELECT")  # force SQL block
    prompt = "\n".join(p)
    return prompt

def extract_sql_from_response(text: str) -> str:
    """
    Try to extract a single SQL statement from model output.
    """
    try:
        t = text or ""
        if "```sql" in t:
            s = t.find("```sql") + len("```sql")
            e = t.find("```", s)
            sql = t[s:e].strip()
        elif "```" in t:
            s = t.find("```") + len("```")
            e = t.find("```", s)
            sql = t[s:e].strip()
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()
        elif "SELECT" in t:
            s = t.find("SELECT")
            sql = t[s:].strip()
        else:
            sql = t.strip()

        # keep up to first semicolon
        if ";" in sql:
            sql = sql.split(";", 1)[0] + ";"
        # one line
        sql = sql.replace("\n", " ").replace("\r", " ").strip()
        # simple guard
        if not sql.lower().startswith("select"):
            return "SELECT 1;"
        return sql
    except Exception:
        return "SELECT 1;"

def run():
    assert INPUT_TEST_SET.exists(), f"Missing test set: {INPUT_TEST_SET}"
    data = json.loads(INPUT_TEST_SET.read_text(encoding="utf-8"))
    print(f"[i] Loaded {len(data)} items from {INPUT_TEST_SET}")

    # চাইলে দ্রুত টেস্টের জন্য আনকমেন্ট করুন:
    # data = data[:5]

    for model_name in MODELS_TO_TEST:
        out_sql = RUN_DIR / f"{OUTPUT_FILE_PREFIX}_{model_name.replace('/', '_')}.sql"
        print(f"\n--- Running CoT for {model_name} (with fallback in helper) ---")
        preds = []

        for i, ex in enumerate(data, 1):
            q = ex["question"]
            schema = ex["schema_str"]
            prompt = create_cot_prompt(q, schema)

            try:
                # helper নিজেই pro -> flash -> flash-latest ট্রাই করবে
                resp_text = gemini_generate_text(
                    prompt,
                    timeout=90,
                    temperature=TEMPERATURE_PLAN,
                )
            except Exception as e:
                print(f"  [ERR] {i}/{len(data)} -> {e}")
                resp_text = "SELECT 1;"

            sql = extract_sql_from_response(resp_text)
            preds.append(sql)

            if i % SLEEP_EVERY == 0:
                print(f"  ... processed {i}, short pause ...")
                time.sleep(1.0)

        with out_sql.open("w", encoding="utf-8") as f:
            for s in preds:
                f.write(s + "\n")

        print(f"[OK] Saved -> {out_sql}")

if __name__ == "__main__":
    run()
