# 3_run_scot_sql.py  (PART 1/3)
# SCoT-SQL: Plan -> Plan-Check -> Final SQL
# Output: runs/dev_30/predicted_scot_models_gemini-2.5-pro.sql

import re, os, json, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils_gemini_rest import gemini_generate_text
import sqlite3

def ask_with_retry(prompt, temperature, max_output_tokens, timeout, tries=2, sleep_sec=0.5):
    last = None
    for _ in range(tries):
        try:
            return gemini_generate_text(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        except Exception as e:
            last = e
            time.sleep(sleep_sec)
    raise last

def try_exec(db_path, sql):
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(sql)
        cur.fetchall()
        cur.close(); con.close()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def build_repair_prompt(schema_str, question, checked_plan, bad_sql, error_msg):
    # UPDATE 1: This function body has been replaced as requested.
    # The code fence for 'Bad SQL' is also properly closed (```)
    # to prevent model confusion, matching the intent of your fix.
    return f"""{schema_str.strip()}

The SQL below failed on SQLite with this error:
ERROR: {error_msg}

Task: Produce a corrected SQL that strictly follows the SCoT-Plan and the schema (no new tables/columns).
Return ONLY the corrected SQL in a fenced block.

Question: {question}

SCoT-Plan:
{checked_plan.strip()}

Bad SQL:
```sql
{bad_sql}
```
"""

PROJ_ROOT = Path(__file__).resolve().parent
RUN_DIR   = PROJ_ROOT / "runs" / "dev_30"
INPUT_TEST_SET = RUN_DIR / "my_test_set.json"
OUTPUT_FILE    = RUN_DIR / "predicted_scot_models_gemini-2.5-pro.sql"

SLEEP_EVERY = 10

FEW_SHOT_PLAN = """\
# SCoT Plan (few-shot)
Input Question: List top 3 highest Rating TV series. List the TV series's Episode and Rating.
Schema (API-Docs):
# TV_series (id, Episode, Air_Date, Rating, Share, 18_49_Rating_Share, Viewers_m, Weekly_Rank, Channel)

SCoT-Plan:
- FROM: TV_series
- WHERE: (none)
- GROUP BY: (none)
- HAVING: (none)
- ORDER BY: Rating DESC
- SELECT: Episode, Rating
- LIMIT: 3

# JOIN + GROUP BY example
Input Question: For each stadium, how many singers performed? Return stadium Name and count, highest first.
Schema (API-Docs):
# stadium (Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)
# singer_in_concert (concert_ID, Singer_ID)
# concert (concert_ID, concert_Name, Theme, Stadium_ID, Year)

SCoT-Plan:
- FROM: stadium AS S JOIN concert AS C ON S.Stadium_ID = C.Stadium_ID
        JOIN singer_in_concert AS SIC ON C.concert_ID = SIC.concert_ID
- WHERE: (none)
- GROUP BY: S.Name
- HAVING: (none)
- ORDER BY: COUNT(*) DESC
- SELECT: S.Name, COUNT(*)
- LIMIT: (none)

# Set operation example
Input Question: Which states appear in both Owners and Professionals?
Schema (API-Docs):
# Owners (owner_id, first_name, last_name, street, city, state, zip_code, email_address, home_phone, cell_number)
# Professionals (professional_id, role_code, first_name, street, city, state, zip_code, last_name, email_address, home_phone, cell_number)

SCoT-Plan:
- FROM: Owners ; Professionals
- WHERE: (none)
- GROUP BY: (none)
- HAVING: (none)
- ORDER BY: (none)
- SELECT: state (intersection)
- LIMIT: (none)
"""

FEW_SHOT_CHECK = """\
# Plan-Check Rules:
- Use only tables/columns that appear in the provided API-Docs.
- If a column is not in schema, replace or drop it; never hallucinate new columns.
- If alias is used, define it in FROM/JOIN.
- Prefer INNER JOIN with explicit join keys; avoid CROSS JOIN.
- If ORDER BY uses aggregate, ensure GROUP BY/HAVING are consistent.
- Keep LIMIT if present in plan.
- SQL-92 only; avoid CTE/window functions.
"""

def build_plan_prompt(schema_str: str, question: str) -> str:
    return f"""{schema_str.strip()}

### Task
Given the schema and question, produce a concise SCoT-Plan describing SQL clauses.

Return EXACTLY this format:

SCoT-Plan:
- FROM: <tables and join keys>
- WHERE: <filters or (none)>
- GROUP BY: <cols or (none)>
- HAVING: <conditions or (none)>
- ORDER BY: <keys or (none)>
- SELECT: <final columns>
- LIMIT: <N or (none)>

{FEW_SHOT_PLAN}

Question: {question}
"""

def build_check_prompt(schema_str: str, plan_text: str) -> str:
    return f"""{schema_str.strip()}

{FEW_SHOT_CHECK}

Given this SCoT-Plan, fix illegal tables/columns or clause conflicts so it strictly matches the schema.

Return only the corrected plan in the exact same format:

{plan_text.strip()}
"""

def build_sql_prompt(schema_str: str, checked_plan: str) -> str:
    return f"""{schema_str.strip()}

Using the SCoT-Plan below, write ONLY the final SQL wrapped in a fenced block:

```sql
SELECT ...
```
Rules:
Return exactly ONE SQL statement ending with ";".

Use only columns that exist in the schema; if a needed field is absent, adapt the query (e.g., use COUNT(*), valid join keys) rather than inventing a column.

Use only columns/tables from the schema.

Define aliases in FROM/JOIN before using.

Prefer INNER JOIN unless filters require LEFT JOIN.

No extra text outside the fenced block.

SCoT-Plan:
{checked_plan.strip()}
"""

### PART 2/3

# 3_run_scot_sql.py  (PART 2/3)

def extract_sql(text: str) -> str:
    """Extract the SQL from a fenced block; fallback to first SELECT...;"""
    t = text or ""
    m = re.search(r"```sql\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
    else:
        m2 = re.search(r"(SELECT\b.*?;)", t, flags=re.DOTALL | re.IGNORECASE)
        sql = (m2.group(1).strip() if m2 else "").strip() or "SELECT 1;"

    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"
    return sql.replace("\n", " ").replace("\r", " ").strip()

def run():
    assert INPUT_TEST_SET.exists(), f"Missing test set: {INPUT_TEST_SET}"
    data = json.loads(INPUT_TEST_SET.read_text(encoding="utf-8"))
    print(f"[i] Loaded {len(data)} items from {INPUT_TEST_SET}")

    preds = []
    for i, ex in enumerate(data, 1):
        schema = ex["schema_str"]
        q      = ex["question"]

        # 1) PLAN  (slightly creative, with retry)
        plan_prompt = build_plan_prompt(schema, q)
        try:
            plan = ask_with_retry(
                 plan_prompt, temperature=0.5, max_output_tokens=800, timeout=90
            )
        except Exception as e:
            print(f"  [Plan ERR] {i}/{len(data)} -> {e}")
            preds.append("SELECT 1;")
            continue

        # 2) PLAN-CHECK  (deterministic, with retry)
        check_prompt = build_check_prompt(schema, plan)
        try:
            checked = ask_with_retry(
                check_prompt, temperature=0.0, max_output_tokens=800, timeout=90
            )
        except Exception as e:
            print(f"  [Check ERR] {i}/{len(data)} -> {e}")
            checked = plan  # fallback

        # 3) FINAL SQL: generate two deterministic candidates; pick the first that executes
        sql_prompt = build_sql_prompt(schema, checked)

        candidates = []
        for _ in range(2):
            sql_text = ask_with_retry(
                sql_prompt, temperature=0.0, max_output_tokens=800, timeout=90
            )
            cand_sql = extract_sql(sql_text)
            candidates.append(cand_sql)

        sql = None
        for cand in candidates:
            ok, _ = try_exec(ex["db_path"], cand)
            if ok:
                sql = cand
                break

        # 4) Execution-guided repair: up to 2 attempts (only if both candidates failed)
        if sql is None:
            rep_prompt = build_repair_prompt(schema, q, checked, candidates[0], "failed to execute")
            try:
                fixed_text = ask_with_retry(
                    rep_prompt, temperature=0.0, max_output_tokens=800, timeout=90
                )
                fixed_sql = extract_sql(fixed_text)
                ok2, _ = try_exec(ex["db_path"], fixed_sql)
                if ok2:
                    sql = fixed_sql
                else:
                    rep_prompt2 = build_repair_prompt(schema, q, checked, fixed_sql, "failed to execute")
                    fixed_text2 = ask_with_retry(
                        rep_prompt2, temperature=0.0, max_output_tokens=800, timeout=90
                    )
                    fixed_sql2 = extract_sql(fixed_text2)
                    ok3, _ = try_exec(ex["db_path"], fixed_sql2)
                    sql = fixed_sql2 if ok3 else "SELECT 1;"
            except Exception:
                sql = "SELECT 1;"

        preds.append(sql) 

        # UPDATE 2: Added progress prints and sleep block as requested
        print(f"  [OK] {i}/{len(data)}")
        if i % SLEEP_EVERY == 0:
            print(f"  ... processed {i}, short pause ...")
            time.sleep(1.0)

    # UPDATE 3: Added final file write block as requested
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for s in preds:
            f.write(s + "\n")
    print(f"[OK] Saved -> {OUTPUT_FILE}")

# 3_run_scot_sql.py  (PART 3/3)

if __name__ == "__main__":
    run()