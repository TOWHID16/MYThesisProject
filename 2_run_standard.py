# 2_run_standard.py
# Standard: no reasoning. Direct SQL only (fenced), temp=0.0
import json, time, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils_gemini_rest import gemini_generate_text

PROJ_ROOT = Path(__file__).resolve().parent
RUN_DIR   = PROJ_ROOT / "runs" / "dev_30"
INPUT_TEST_SET = RUN_DIR / "my_test_set.json"
OUTPUT_FILE    = RUN_DIR / "predicted_standard.sql"
SLEEP_EVERY = 10

def build_prompt(schema_str: str, question: str) -> str:
    # (A) UPDATE: The prompt format has been updated as per your instructions.
    # The "SELECT ..." part has been moved out of the example.
    return f"""{schema_str.strip()}

Return ONLY the final SQL for the question below, wrapped in a fenced block like:

```sql
SELECT ...
```
Rules:

Use only the tables/columns that appear in the schema above.
Define aliases in FROM/JOIN before using them.
SQL-92 only (no CTE/window).
No explanation outside the fenced block.
Question: {question}

"""

def extract_sql(text: str) -> str:
    # (B) UPDATE: The Regex logic has been updated as per your instructions.
    t = text or ""
    
    # 1) First, search for the complete fenced block (```sql ... ```).
    m = re.search(r"```sql\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    
    if m:
        sql = m.group(1).strip()
    else:
        # 2) If no fenced block is found, fallback to searching for a plain "SELECT ... ;".
        m2 = re.search(r"(SELECT\b.*?;)", t, flags=re.DOTALL | re.IGNORECASE)
        sql = (m2.group(1).strip() if m2 else "SELECT 1;")

    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"
        
    return sql.replace("\n", " ").replace("\r", " ").strip()

def main():
    data = json.loads(INPUT_TEST_SET.read_text(encoding="utf-8"))
    print(f"[i] Loaded {len(data)} items")
    preds = []
    for i, ex in enumerate(data, 1):
        p = build_prompt(ex["schema_str"], ex["question"])
        try:
            txt = gemini_generate_text(p, temperature=0.0, max_output_tokens=600, timeout=90)
            sql = extract_sql(txt)
        except Exception as e:
            print(f" [ERR] {i}/{len(data)} -> {e}")
            sql = "SELECT 1;"
        preds.append(sql)
        if i % SLEEP_EVERY == 0:
            print(f" ... processed {i}, pause ...")
            time.sleep(1.0)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for s in preds: f.write(s + "\n")
    print(f"[OK] Saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()