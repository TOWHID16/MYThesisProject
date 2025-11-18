# 2_run_qdecomp_intercol.py
# QDecomp+InterCOL: each step lists used table.column; then final SQL
import json, time, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils_gemini_rest import gemini_generate_text

PROJ_ROOT = Path(__file__).resolve().parent
RUN_DIR   = PROJ_ROOT / "runs" / "dev_600"
INPUT_TEST_SET = RUN_DIR / "my_test_set.json"
OUTPUT_FILE    = RUN_DIR / "predicted_qd_intercol.sql"
SLEEP_EVERY = 10

FEW_SHOT = """\
Example:

Schema:
# TV_series (id, Episode, Air_Date, Rating, Share, 18_49_Rating_Share, Viewers_m, Weekly_Rank, Channel)

Question: Top 3 series by Rating; return Episode, Rating.

Decomposition (InterCOL):
1) Choose table -> TV_series     [Cols: TV_series.Episode, TV_series.Rating]
2) Sort by Rating DESC         [Cols: TV_series.Rating]
3) Limit 3                     [Cols: (none)]
4) Select Episode, Rating       [Cols: TV_series.Episode, TV_series.Rating]

Final SQL:
```sql
SELECT Episode, Rating
FROM TV_series
ORDER BY Rating DESC
LIMIT 3;
```

"""

def build_prompt(schema_str: str, question: str) -> str:
    # (A) UPDATE: The prompt format has been updated as per your instructions.
    # The "Format:" section is more explicit, and Rules are included.
    return f"""{schema_str.strip()}

Decompose into 3–8 steps. After EVERY step, add InterCOL bracket listing the exact table.column used.
Then output ONLY the final SQL in a fenced block:

Format:
Decomposition (InterCOL):
- ... [Cols: table.col, table.col]
- ... [Cols: ...]
Final SQL:
```sql
SELECT ...
```
Rules:

Use only tables/columns present in schema.
Define aliases before using.
SQL-92 only; no CTE/window.
{FEW_SHOT}
Question: {question}

"""

def extract_sql(text: str) -> str:
    # (B) UPDATE: The Regex logic has been updated (same as the other scripts).
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
            txt = gemini_generate_text(p, temperature=0.2, max_output_tokens=900, timeout=90)
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