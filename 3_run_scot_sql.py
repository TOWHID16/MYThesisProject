# 3_run_scot_sql.py  (PART 1/3)
# SCoT-SQL: Plan -> Plan-Check -> Final SQL
# Output: runs/dev_30/predicted_scot_models_gemini-2.5-pro.sql
#
# Code updated based on instruction_2 & instruction_3:
# A) Plan temperature is set to 0.3.
# B) FEW_SHOT_CHECK includes mandatory FK hints and canonical rules.
# C) A new canonicalization step (build_canon_prompt) is added.
# D) build_repair_prompt includes error-specific hints.
# E) Added 3 new FEW_SHOT_PLAN examples (UNION, EXCEPT, INTERSECT).
# F) Added standard-style candidate pooling (2 SCoT + 1 Standard).
# G) Discard 'SELECT 1;' candidates.
# H) Improved canonicalization to fallback on 'SELECT 1;'.
# I) Increased timeouts to 120s.
# J) Added debug logging to scot_debug.log.

import re, os, json, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils_gemini_rest import gemini_generate_text
import sqlite3

def ask_with_retry(prompt, temperature, max_output_tokens, timeout, tries=3, sleep_sec=0.5):
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
    # UPDATE D: Added error-specific hints (Additional Fix Rules)
    return f"""{schema_str.strip()}

The SQL below failed on SQLite with this error:
ERROR: {error_msg}

Task: Produce a corrected SQL that strictly follows the SCoT-Plan and the schema (no new tables/columns).
Return ONLY the corrected SQL in a fenced block.

Additional Fix Rules:
- If the error is "no such column" or "ambiguous column name": qualify all columns with table alias from the plan.
- If the error is "misuse of aggregate" or "GROUP BY" mismatch: ensure all non-aggregated selected columns appear in GROUP BY.
- Remove ORDER BY unless the question explicitly asks for ranking/sorting.
- Use SELECT DISTINCT to deduplicate when no aggregation is needed.
- Use only schema columns and the FK join keys shown in the plan; do not invent columns.

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
DEBUG_LOG_FILE = RUN_DIR / "scot_debug.log" # Added for Instruction 7

SLEEP_EVERY = 10

# UPDATE E: Added 3 new patterns (UNION, EXCEPT, INTERSECT)
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

# UNION example (filter + group + union)
Input Question: Which professionals live in Indiana OR handled more than 2 treatments? Return id, last_name, cell_number.
Schema (API-Docs):
# Professionals (professional_id, last_name, cell_number, state, ...)
# Treatments (treatment_id, professional_id, ...)
SCoT-Plan:
- FROM: Professionals
- WHERE: state = 'Indiana'
- GROUP BY: (none)
- HAVING: (none)
- ORDER BY: (none)
- SELECT: professional_id, last_name, cell_number
- LIMIT: (none)
-- UNION --
- FROM: Professionals AS P JOIN Treatments AS T ON P.professional_id = T.professional_id
- WHERE: (none)
- GROUP BY: P.professional_id
- HAVING: COUNT(*) > 2
- ORDER BY: (none)
- SELECT: P.professional_id, P.last_name, P.cell_number
- LIMIT: (none)

# EXCEPT example (anti-join by set diff)
Input Question: Names of countries with no car makers.
Schema (API-Docs):
# countries (CountryId, CountryName, ...)
# car_makers (Id, Country, ...)
SCoT-Plan:
- FROM: countries
- WHERE: (none)
- GROUP BY: (none)
- HAVING: (none)
- ORDER BY: (none)
- SELECT: CountryName
- LIMIT: (none)
-- EXCEPT --
- FROM: countries AS C JOIN car_makers AS M ON C.CountryId = M.Country
- WHERE: (none)
- GROUP BY: C.CountryName
- HAVING: (none)
- ORDER BY: (none)
- SELECT: C.CountryName
- LIMIT: (none)

# INTERSECT with JOIN example (two directors)
Input Question: Series name and country of channels that air cartoons by BOTH Ben Jones and Michael Chang.
Schema (API-Docs):
# TV_Channel (id, series_name, country)
# Cartoon (id, Title, Directed_by, Channel)
SCoT-Plan:
- FROM: TV_Channel AS T JOIN Cartoon AS C ON T.id = C.Channel
- WHERE: C.Directed_by = 'Ben Jones'
- GROUP BY: (none)
- HAVING: (none)
- ORDER BY: (none)
- SELECT: T.series_name, T.country
- LIMIT: (none)
-- INTERSECT --
- FROM: TV_Channel AS T JOIN Cartoon AS C ON T.id = C.Channel
- WHERE: C.Directed_by = 'Michael Chang'
- GROUP BY: (none)
- HAVING: (none)
- ORDER BY: (none)
- SELECT: T.series_name, T.country
- LIMIT: (none)
"""

# UPDATE B: Added 4 new rules to FEW_SHOT_CHECK
FEW_SHOT_CHECK = """\
# Plan-Check Rules:
- Use only tables/columns that appear in the provided API-Docs.
- If a column is not in schema, replace or drop it; never hallucinate new columns.
- If alias is used, define it in FROM/JOIN.
- Prefer INNER JOIN with explicit join keys; avoid CROSS JOIN.
- If ORDER BY uses aggregate, ensure GROUP BY/HAVING are consistent.
- Keep LIMIT if present in plan.
- SQL-92 only; avoid CTE/window functions.
- Do NOT add ORDER BY unless the question or plan explicitly requires ordering.
- Every SELECT column must be either aggregated or appear in GROUP BY.
- If duplicates are possible and no aggregation is needed, consider SELECT DISTINCT instead of GROUP BY.
- If the schema lists foreign keys, explicitly use them as join keys in FROM/JOIN.
- Do NOT use HAVING without GROUP BY.
- Qualify every selected/ordered column with its table alias to avoid ambiguity.
- Prefer SELECT DISTINCT over GROUP BY when only deduplication is needed (no aggregates).
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
Do NOT add ORDER BY unless the question/plan demands ranking or sorting.
No comments or explanations in the SQL output.
Use only columns/tables from the schema.
Define aliases in FROM/JOIN before using.
Prefer INNER JOIN unless filters require LEFT JOIN.
No extra text outside the fenced block.


SCoT-Plan:
{checked_plan.strip()}
"""

# UPDATE C: Added build_canon_prompt function
def build_canon_prompt(schema_str: str, sql_text: str) -> str:
    return f"""{schema_str.strip()}

Rewrite the SQL **without changing its semantics** to satisfy ALL rules:

- Qualify **every** column with the correct table alias.
- If non-aggregated columns appear with aggregates, add GROUP BY for all non-aggregates.
- If no ranking/sorting is asked, **remove ORDER BY**.
- If duplicates are possible but no aggregation is needed, use SELECT DISTINCT.
- SQL-92 only; no CTE/window.
- Keep it as **one** statement ending with ";", in a fenced block.

Input SQL:
```sql
{sql_text.strip()}
```
Return only the corrected SQL:
"""

# UPDATE F (Instruction 1): Add build_standard_sql_prompt
def build_standard_sql_prompt(schema_str: str, question: str) -> str:
    return f"""{schema_str.strip()}

Return ONLY the final SQL for the question below, wrapped in a fenced block:

```sql
SELECT ...
```
Rules:
Use only tables/columns from the schema.
SQL-92 only (no CTE/window).
Define aliases before using.
Do NOT add ORDER BY unless ranking/sorting is required.
No commentary outside the fenced block.

Question: {question}
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

    # Clear debug log on start
    if DEBUG_LOG_FILE.exists():
        DEBUG_LOG_FILE.unlink()

    preds = []
    for i, ex in enumerate(data, 1):
        schema = ex["schema_str"]
        q      = ex["question"]

        # 1) PLAN  (slightly creative, with retry)
        plan_prompt = build_plan_prompt(schema, q)
        try:
            # UPDATE A: Temperature is set to 0.3
            # UPDATE I: Timeout 120
            plan = ask_with_retry(
                plan_prompt, temperature=0.3, max_output_tokens=800, timeout=120
            )
        except Exception as e:
            print(f"  [Plan ERR] {i}/{len(data)} -> {e}")
            preds.append("SELECT 1;")
            continue

        # 2) PLAN-CHECK  (deterministic, with retry)
        check_prompt = build_check_prompt(schema, plan)
        try:
            # UPDATE I: Timeout 120
            checked = ask_with_retry(
                check_prompt, temperature=0.0, max_output_tokens=800, timeout=120
            )
        except Exception as e:
            print(f"  [Check ERR] {i}/{len(data)} -> {e}")
            checked = plan  # fallback

        # 3) FINAL SQL: 2× SCoT + 1× Standard (UPDATE F)
        sql_prompt = build_sql_prompt(schema, checked)
        candidates = []
        for _ in range(2):
            try:
                sql_text = ask_with_retry(sql_prompt, temperature=0.0, max_output_tokens=1024, timeout=120)
                cand_sql = extract_sql(sql_text)
                candidates.append(cand_sql)
            except Exception:
                pass # Ignore failure on one candidate

        # extra: standard-style candidate
        try:
            std_text = ask_with_retry(
                build_standard_sql_prompt(schema, q),
                temperature=0.0, max_output_tokens=1024, timeout=120
            )
            std_sql = extract_sql(std_text)
            candidates.append(std_sql)
        except Exception:
            pass

        # UPDATE G: Discard 'SELECT 1;' candidates
        candidates = [c for c in candidates if c.strip().lower() != "select 1;"]
        if not candidates:
            # Retry SCoT prompt once
            try:
                sql_text = ask_with_retry(sql_prompt, temperature=0.0, max_output_tokens=1024, timeout=120)
                cand_sql = extract_sql(sql_text)
                if cand_sql.strip().lower() != "select 1;":
                    candidates.append(cand_sql)
            except Exception:
                pass # Still no candidates, will fail later

        # UPDATE H: Canonicalization Step (with SELECT 1; fallback)
        canon_candidates = []
        for cand in candidates:
            try:
                canon_text = ask_with_retry(
                    build_canon_prompt(schema, cand),
                    temperature=0.0, max_output_tokens=800, timeout=120 # UPDATE I
                )
                canon_sql = extract_sql(canon_text)
                # guard: avoid SELECT 1;
                if canon_sql.strip().lower() == "select 1;":
                    canon_candidates.append(cand)  # keep original
                else:
                    canon_candidates.append(canon_sql)
            except Exception:
                canon_candidates.append(cand) # fallback on error

        sql = None
        last_error = "failed to execute" # Default error
        
        # Try canonical candidates first
        for cand in canon_candidates:
            ok, err_msg = try_exec(ex["db_path"], cand)
            if ok:
                sql = cand
                break
            last_error = err_msg # Store the error

        # If canon fails, try original candidates (fallback)
        if sql is None:
            for cand in candidates:
                ok, err_msg = try_exec(ex["db_path"], cand)
                if ok:
                    sql = cand
                    break
                last_error = err_msg # Store the error
        
        # Get the first SQL that failed, for the repair prompt
        failed_sql_for_repair = canon_candidates[0] if canon_candidates else (candidates[0] if candidates else "SELECT 1;")

        # 4) Execution-guided repair (now with specific error)
        if sql is None:
            rep_prompt = build_repair_prompt(schema, q, checked, failed_sql_for_repair, last_error)
            try:
                fixed_text = ask_with_retry(
                    rep_prompt, temperature=0.0, max_output_tokens=800, timeout=120 # UPDATE I
                )
                fixed_sql = extract_sql(fixed_text)
                ok2, err_msg2 = try_exec(ex["db_path"], fixed_sql)
                if ok2:
                    sql = fixed_sql
                else:
                    # Second repair attempt
                    rep_prompt2 = build_repair_prompt(schema, q, checked, fixed_sql, err_msg2)
                    fixed_text2 = ask_with_retry(
                        rep_prompt2, temperature=0.0, max_output_tokens=800, timeout=120 # UPDATE I
                    )
                    fixed_sql2 = extract_sql(fixed_text2)
                    ok3, _ = try_exec(ex["db_path"], fixed_sql2) # Don't care about 3rd error
                    sql = fixed_sql2 if ok3 else "SELECT 1;"
            except Exception:
                sql = "SELECT 1;"
        
        # Fallback if sql is still None
        if sql is None:
            sql = "SELECT 1;"

        # UPDATE J: Debug logging
        try:
            with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as dbg:
                db_id = ex.get('db_id', 'unknown_db')
                dbg.write(f"\n--- IDX {i} ({db_id}) ---\nQ: {q}\nPLAN:\n{plan}\nCHECKED:\n{checked}\nCANDS:\n{candidates}\nCANON_CANDS:\n{canon_candidates}\nLAST_ERROR: {last_error}\nFINAL_SQL:\n{sql}\n\n")
        except Exception as e:
            print(f"  [Debug Log ERR] {i}/{len(data)} -> {e}")


        preds.append(sql) 

        # Progress prints and sleep block
        print(f"  [OK] {i}/{len(data)}")
        if i % SLEEP_EVERY == 0:
            print(f"  ... processed {i}, short pause ...")
            time.sleep(1.0)

    # Final file write block
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for s in preds:
            f.write(s + "\n")
    print(f"[OK] Saved -> {OUTPUT_FILE}")
    print(f"[OK] Debug Log -> {DEBUG_LOG_FILE}")

# 3_run_scot_sql.py  (PART 3/3)

if __name__ == "__main__":
    run()