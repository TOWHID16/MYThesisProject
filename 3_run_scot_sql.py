# SCoT-SQL: Plan -> Plan-Check -> Final SQL
# Output: runs/dev_30/predicted_scot_models_gemini-2.5-pro.sql
# Code updated based on instruction_2, instruction_3, & instruction_4:
# A) Plan temperature is set to 0.3.
# B) FEW_SHOT_CHECK includes mandatory FK hints and canonical rules.
# C) A new canonicalization step (build_canon_prompt) is added.
# D) build_repair_prompt includes error-specific hints.
# E) Added 3 new FEW_SHOT_PLAN examples (UNION, EXCEPT, INTERSECT).
# F) Added standard-style candidate pooling (Now 3 SCoT + 1 Standard).
# G) Discard 'SELECT 1;' candidates.
# H) Improved canonicalization to fallback on 'SELECT 1;'.
# I) Increased timeouts to 120s.
# J) Added debug logging to scot_debug.log.
# K) (instr_4) Updated repair/canon prompts to preserve ORDER BY.
# L) (instr_4) Added plan-guard helpers (_enforce_plan, _add_nocase) for post-processing.

import re, os, json, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils_gemini_rest import gemini_generate_text
import sqlite3
import typing as _t

MAX_SQL_LEN = 4000  # sanity guard for oversized generations

# UPDATE D (instr_4): Add Plan-guard + NOCASE heuristic helpers
def _parse_plan(checked_plan: str) -> dict:
    """Parse minimal ORDER BY and LIMIT from the checked plan."""
    ob = None
    lim = None
    for line in checked_plan.splitlines():
        line = line.strip()
        if line.lower().startswith("- order by:"):
            ob = line.split(":",1)[1].strip()
        if line.lower().startswith("- limit:"):
            lim = line.split(":",1)[1].strip()
    return {"order_by": ob, "limit": lim}

def _add_nocase(sql: str) -> str:
    """Make string equality case-insensitive via COLLATE NOCASE."""
    # ... alias.col = 'text'  -> alias.col COLLATE NOCASE = 'text'
    return re.sub(r"(\b\w+\.\w+\b)\s*=\s*('(?:[^']|'')*')", r"\1 COLLATE NOCASE = \2", sql, flags=re.I)

def _normalize_order_count(expr: str) -> str:
    # ORDER BY COUNT(anything) -> ORDER BY COUNT(*)
    return re.sub(r"COUNT\s*\([^)]*\)", "COUNT(*)", expr, flags=re.I)

def _ensure_semicolon(s: str) -> str:
    s = s.strip()
    if not s.endswith(";"):
        s += ";"
    return s

def _enforce_plan(sql: str, checked_plan: str) -> str:
    """Append ORDER BY/LIMIT from plan if missing; keep existing ones."""
    sql0 = sql.strip().rstrip(";")
    meta = _parse_plan(checked_plan)
    order_by = (meta.get("order_by") or "").strip()
    limit    = (meta.get("limit") or "").strip()

    out = sql0

    # Add ORDER BY if plan has it and SQL doesn't
    if order_by and order_by.lower() != "(none)" and " order by " not in out.lower():
        # Normalize common COUNT(...) DESC
        ob = _normalize_order_count(order_by)
        out += f" ORDER BY {ob}"

    # Add LIMIT if plan has it and SQL doesn't
    if limit and limit.lower() != "(none)" and " limit " not in out.lower():
        # LIMIT could be like "1" or "3"
        m = re.match(r"(\d+)", limit)
        if m:
            out += f" LIMIT {m.group(1)}"
    return _ensure_semicolon(out)

def _segment_plan(plan_text: str) -> dict:
    """Split SCoT-Plan text into clause->lines (first match wins for each clause)."""
    clauses = ["- from:", "- where:", "- group by:", "- having:", "- order by:", "- select:", "- limit:"]
    out = {c: [] for c in clauses}
    for line in plan_text.splitlines():
        s = line.strip()
        sl = s.lower()
        for c in clauses:
            if sl.startswith(c):
                out[c].append(s.split(":", 1)[1].strip())
                break
    return out

def _dedup_clause_block(vals: list[str]) -> str:
    """Keep first non-empty; if multiple exist, prefer the first unique."""
    if not vals:
        return "(none)"
    seen = set()
    for v in vals:
        vv = (v or "").strip()
        if not vv:
            continue
        if vv.lower() not in seen:
            seen.add(vv.lower())
            return vv
    return "(none)"

def _normalize_plan(plan_text: str) -> str:
    """Normalize duplicated clauses & rebuild the plan in canonical order."""
    if not plan_text or "SCoT-Plan:" not in plan_text:
        return plan_text
    blocks = _segment_plan(plan_text)
    fmt = [
        ("- FROM:",      _dedup_clause_block(blocks.get("- from:",      []))),
        ("- WHERE:",     _dedup_clause_block(blocks.get("- where:",     []))),
        ("- GROUP BY:",  _dedup_clause_block(blocks.get("- group by:",  []))),
        ("- HAVING:",    _dedup_clause_block(blocks.get("- having:",    []))),
        ("- ORDER BY:",  _dedup_clause_block(blocks.get("- order by:",  []))),
        ("- SELECT:",    _dedup_clause_block(blocks.get("- select:",    []))),
        ("- LIMIT:",     _dedup_clause_block(blocks.get("- limit:",     []))),
    ]
    lines = ["SCoT-Plan:"] + [f"{k} {v}" for k, v in fmt]
    return "\n".join(lines)

# === Enforce plan table coverage on candidates ===
_NEG_TOKENS = {
    "no", "without", "none", "missing", "lack", "exclude", "not present",
    "neither", "except those", "no such", "no one", "no country", "no maker",
    "নেই", "ছাড়া", "বিহীন", "শূন্য", "অনুপস্থিত"
}

def _needs_setdiff(q: str) -> bool:
    if not q:
        return False
    s = q.strip().lower()
    return any(tok in s for tok in _NEG_TOKENS)

def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        k = (x or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

# === SQL truncation detection ===
_TRUNC_PATTERNS = (
    r"\bON\s*$", r"\bWHERE\s*$", r"\bAND\s*$", r"\bOR\s*$",
    r"=\s*$", r",\s*$", r"\bJOIN\s*$", r"\bGROUP BY\s*$",
    r"\bHAVING\s*$", r"\bORDER BY\s*$", r"\bUNION\s*$",
    r"\bEXCEPT\s*$", r"\bINTERSECT\s*$"
)

def _is_viable_sql(sql: str) -> bool:
    s = (sql or "").strip().lower()
    if not s.startswith("select"):
        return False
    if (" from " not in s) and not any(op in s for op in (" union ", " except ", " intersect ")):
        return False
    # Paren balance
    if s.count("(") != s.count(")"):
        return False
    # Ends with suspicious dangling tokens?
    for pat in _TRUNC_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return False
    return True

def _clean_bad_tail(sql: str) -> str:
    """Trim dangling fragments at the end; keep last safe token boundary."""
    s = (sql or "").strip()
    # Try to cut back to last complete clause terminator
    cutpoints = [" UNION ", " EXCEPT ", " INTERSECT ", " ORDER BY ", " HAVING ", " GROUP BY ", " WHERE ", " JOIN ", " FROM "]
    last = -1
    for tok in cutpoints:
        p = s.upper().rfind(tok)
        last = max(last, p)
    if last > 0:
        s = s[:last].strip()
    # Ensure it still ends with ';'
    if not s.endswith(";"):
        s += ";"
    return s

def _prefer_setdiff(sql: str, question: str) -> str:
    if not _needs_setdiff(question):
        return sql
    s = sql.lower()
    # If it already uses EXCEPT / anti-join filter, leave it
    if " except " in s or (" left join " in s and " is null" in s):
        return sql
    # Light-touch heuristic: if it's a LEFT JOIN without the IS NULL anti-filter, add it when safe.
    # This avoids hallucination: only add when there's exactly one JOIN and one right alias.
    m = re.search(r"\bfrom\s+([a-z_][a-z0-9_]*)\s+as\s+([a-z]\w*)\b.*?\bleft\s+join\s+([a-z_][a-z0-9_]*)\s+as\s+([a-z]\w*)\b", sql, flags=re.I|re.S)
    if m and " where " not in s:
        right_alias = m.group(4)
        return sql.rstrip(";") + f" WHERE {right_alias}.rowid IS NULL;"
    return sql

# === Schema validation ===
def _collect_schema_symbols(schema_str: str) -> tuple[set[str], set[str]]:
    """
    Returns (tables, columns) in lowercase.
    Accepts lines like: "# table_name (colA, colB, ...)" (your schema format).
    """
    tbls, cols = set(), set()
    for line in schema_str.splitlines():
        line = line.strip()
        if not line.startswith("# "): 
            continue
        if "(" in line and ")" in line:
            # "# table (a, b, c)"
            head, inside = line[2:].split("(", 1)
            t = head.strip().split()[0]
            tbls.add(t.lower())
            for c in inside.split(")")[0].split(","):
                cc = c.strip()
                if cc:
                    cols.add(cc.split()[0].lower())
    return tbls, cols

_COLREF = re.compile(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", re.I)
_TBLREF = re.compile(r"\bfrom\s+([a-z_][a-z0-9_]*)\b|\bjoin\s+([a-z_][a-z0-9_]*)\b", re.I)

def _uses_only_schema(sql: str, schema_tables: set[str], schema_cols: set[str]) -> bool:
    # table names (FROM/JOIN)
    for m in _TBLREF.finditer(sql):
        t = (m.group(1) or m.group(2) or "").lower()
        if t and t not in schema_tables:
            return False
    # qualified col refs
    for m in _COLREF.finditer(sql):
        col = (m.group(2) or "").lower()
        if col and col not in schema_cols:
            return False
    return True

def _score_against_plan(sql: str, checked_plan: str) -> int:
    s = sql.lower()
    p = checked_plan.lower()
    score = 0
    if "- order by:" in p:
        ob = p.split("- order by:",1)[1].split("\n",1)[0].strip()
        if ob != "(none)" and " order by " in s: score += 2
        if ob == "(none)" and " order by " not in s: score += 1
        # Bonus for matching COUNT(*) in ORDER BY
        if "count(" in ob and "order by" in s and "count(*)" in s:
            score += 1
    if "- limit:" in p:
        lim = p.split("- limit:",1)[1].split("\n",1)[0].strip()
        if lim != "(none)" and " limit " in s: score += 2
        if lim == "(none)" and " limit " not in s: score += 1
    # encourage plan SELECT tokens presence (very light heuristic)
    if "- select:" in p:
        want = p.split("- select:",1)[1].split("\n",1)[0]
        for tok in re.findall(r"[a-z_][a-z0-9_]*", want, flags=re.I):
            if tok and tok in s: score += 1
    return score

# === Enforce plan table coverage on candidates ===
def _tables_in_plan(checked_plan: str) -> set[str]:
    STOP = {
        "join","on","left","right","inner","outer","as",
        "union","intersect","except","where","group","having",
        "order","select","limit"
    }
    want = set()
    for line in checked_plan.splitlines():
        if line.strip().lower().startswith("- from:"):
            frag = line.split(":", 1)[1].lower()
            for t in re.findall(r"\b([a-z_][a-z0-9_]*)\b(?:\s+as\s+[a-z_][a-z0-9_]*)?", frag):
                if t not in STOP:
                    want.add(t)
    return want

def _tables_in_sql(sql: str) -> set[str]:
    s = sql.lower()
    tbls = set(re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)\b", s))
    tbls |= set(re.findall(r"\bjoin\s+([a-z_][a-z0-9_]*)\b", s))
    return tbls

def _filter_by_plan_tables(cands: list[str], checked_plan: str) -> list[str]:
    need = _tables_in_plan(checked_plan)
    if not need:
        return cands
    good, weak = [], []
    for c in cands:
        have = _tables_in_sql(c)
        (good if need.issubset(have) else weak).append(c)
    return good + weak

# === Normalize model responses to plain text (handles dict/JSON) ===
def _to_text(x):
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

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

_GEN_CACHE = {}
def ask_with_retry_cached(prompt, temperature, max_output_tokens, timeout, tries=3, sleep_sec=0.5):
    k = (prompt, temperature, max_output_tokens)
    if k in _GEN_CACHE: 
        return _GEN_CACHE[k]
    out = ask_with_retry(prompt, temperature, max_output_tokens, timeout, tries, sleep_sec)
    _GEN_CACHE[k] = out
    return out

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
    # UPDATE B (instr_4): Preserve ORDER BY rule
    # Add negation-aware hint automatically
    extra_hint = ""
    try:
        if _needs_setdiff(question):
            extra_hint = "- Since the question implies a negation (no/without), prefer EXCEPT or LEFT JOIN ... WHERE right.id IS NULL for set-difference.\n"
    except Exception:
        pass

    return f"""{schema_str.strip()}

The SQL below failed on SQLite with this error:
ERROR: {error_msg}

Task: Produce a corrected SQL that strictly follows the SCoT-Plan and the schema (no new tables/columns).
Return ONLY the corrected SQL in a fenced block.

Additional Fix Rules:
- If the error is "no such column" or "ambiguous column name": qualify all columns with table alias from the plan.
- If the error is "misuse of aggregate" or "GROUP BY" mismatch: ensure all non-aggregated selected columns appear in GROUP BY.
- Preserve ORDER BY and LIMIT exactly as specified by the SCoT-Plan. Only omit them if the plan shows (none).
- Use SELECT DISTINCT to deduplicate when no aggregation is needed.
- Use only schema columns and the FK join keys shown in the plan; do not invent columns.
{extra_hint}Question: {question}

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
FEW_SHOT_PLAN = """# SCoT Plan (few-shot)
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
FEW_SHOT_CHECK = """# Plan-Check Rules:
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
# Rules for popularity wording:
# - If the question says "most popular" without mentioning "percentage/share", treat popularity as COUNT of entities (e.g., number of countries).
# - Only use SUM(Percentage) when the question explicitly asks by percentage/share.

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
    # UPDATE C (instr_4): Preserve ORDER BY rule
    return f"""{schema_str.strip()}

Rewrite the SQL **without changing its semantics** to satisfy ALL rules:

- Qualify **every** column with the correct table alias.
- If non-aggregated columns appear with aggregates, add GROUP BY for all non-aggregates.
- Do **not** remove ORDER BY if it is present; preserve existing ORDER BY/LIMIT (do not add new ones).
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

def build_standard_sql_prompt_setdiff(schema_str: str, question: str) -> str:
    """Bias toward EXCEPT/anti-join when the question implies negation."""
    return f"""{schema_str.strip()}

Return ONLY the final SQL for the question below, wrapped in a fenced block:

```sql
SELECT ...
```
Rules:
Prefer a set-difference approach (EXCEPT) or anti-join (LEFT JOIN ... WHERE right.id IS NULL) when the question asks for items with none/without.
Use only tables/columns from the schema.
SQL-92 only (no CTE/window).
Define aliases before using.
Do NOT add ORDER BY unless ranking/sorting is required.
No commentary outside the fenced block.

Question: {question}
"""


def extract_sql(text: str) -> str:
    """Extract the SQL from a fenced block; fallback to first SELECT...;"""
    t = text or ""

    # 1) Prefer code fence (with or without 'sql' tag)
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        sql = (m.group(1) or "").strip()
    else:
        sql = ""

    # 2) If fence was missing or empty, fall back to first SELECT...;
    if not sql:
        m2 = re.search(r"(SELECT\b.*?;)", t, flags=re.DOTALL | re.IGNORECASE)
        sql = (m2.group(1).strip() if m2 else "SELECT 1;")

    # 3) Normalize to a single statement, keep trailing semicolon
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
        
        # Collect schema symbols for validation
        schema_tables, schema_cols = _collect_schema_symbols(schema)

        # 1) PLAN  (slightly creative, with retry)
        plan_prompt = build_plan_prompt(schema, q)
        try:
            # UPDATE A: Temperature is set to 0.3
            # UPDATE I: Timeout 120
            plan = _to_text(ask_with_retry_cached(
    plan_prompt, temperature=0.3, max_output_tokens=800, timeout=120
))

        except Exception as e:
            print(f"  [Plan ERR] {i}/{len(data)} -> {e}")
            preds.append("SELECT 1;")
            continue

        # 2) PLAN-CHECK  (deterministic, with retry)
        check_prompt = build_check_prompt(schema, plan)
        try:
            checked = _to_text(ask_with_retry_cached(
                check_prompt, temperature=0.0, max_output_tokens=800, timeout=120
            ))
        except Exception as e:
            print(f"  [Check ERR] {i}/{len(data)} -> {e}")
            checked = plan  # fallback
        
        # Normalize plan to dedupe repeated clauses
        checked = _normalize_plan(checked)
        
        # Sanity: model যদি JSON/garbage দেয়
        if "SCoT-Plan:" not in checked:
            checked = plan
        
        # 3) FINAL SQL: 3x SCoT + 1x Standard
        sql_prompt = build_sql_prompt(schema, checked)
        candidates = []
        raw_sql_texts = []  # Capture raw model outputs
        # UPDATE A (instr_4): Change 2 -> 3
        for _ in range(3):
            try:
                sql_text = _to_text(ask_with_retry_cached(sql_prompt, temperature=0.0, max_output_tokens=1024, timeout=120))
                raw_sql_texts.append(sql_text)

                cand_sql = extract_sql(sql_text)
                candidates.append(cand_sql)
            except Exception:
                pass # Ignore failure on one candidate

        # extra: standard-style candidate
        try:
            std_text = _to_text(ask_with_retry_cached(
    build_standard_sql_prompt(schema, q),
    temperature=0.0, max_output_tokens=1024, timeout=120
))
            raw_sql_texts.append(std_text)

            std_sql = extract_sql(std_text)
            candidates.append(std_sql)
        except Exception:
            pass

        # Optional: negation-aware extra candidate
        try:
            if _needs_setdiff(q):
                std_neg_text = _to_text(ask_with_retry_cached(
                    build_standard_sql_prompt_setdiff(schema, q),
                    temperature=0.0, max_output_tokens=1024, timeout=120
                ))
                raw_sql_texts.append(std_neg_text)
                std_neg_sql = extract_sql(std_neg_text)
                candidates.append(std_neg_sql)
        except Exception:
            pass

        # Drop trivial fallbacks
        candidates = [c for c in candidates if c.strip().lower() != "select 1;"]
        
        # Length guard
        candidates = [c[:MAX_SQL_LEN] for c in candidates]
        # Deduplicate while preserving order
        candidates = _dedup_preserve_order(candidates)
        
        # Filter out truncated/broken SQL
        candidates = [c for c in candidates if _is_viable_sql(c)]
        # Try to clean bad tails
        candidates = [c if _is_viable_sql(c) else _clean_bad_tail(c) for c in candidates]
        
        # Schema validation: reject candidates with invalid table/column refs
        candidates = [c for c in candidates if _uses_only_schema(c, schema_tables, schema_cols)]

        # Retry once if we lost all candidates
        if not candidates:
            try:
                sql_text = _to_text(ask_with_retry_cached(sql_prompt, temperature=0.0, max_output_tokens=1024, timeout=120))
                cand_sql = extract_sql(sql_text)
                if cand_sql.strip().lower() != "select 1;":
                    candidates.append(cand_sql)
            except Exception:
                pass
        
        # ---- Canonicalization step ----
        canon_candidates = []
        canon_raw_texts = []  # Capture raw canon outputs
        for cand in candidates:
            try:
                canon_text = _to_text(ask_with_retry_cached(
                    build_canon_prompt(schema, cand),
                    temperature=0.0, max_output_tokens=800, timeout=120
                ))
                canon_raw_texts.append(canon_text)
                canon_sql = extract_sql(canon_text)
                canon_candidates.append(canon_sql if canon_sql.strip().lower() != "select 1;" else cand)
            except Exception:
                canon_candidates.append(cand)  # fallback on error
        
        # ---- Post-process (NOCASE + enforce plan ORDER/LIMIT) ----
        canon_candidates = [_enforce_plan(_add_nocase(c), checked) for c in canon_candidates]
        candidates       = [_enforce_plan(_add_nocase(c), checked) for c in candidates]
        
        # Apply setdiff preference for negation queries
        if _needs_setdiff(q):
            canon_candidates = [_prefer_setdiff(c, q) for c in canon_candidates]
            candidates       = [_prefer_setdiff(c, q) for c in candidates]
        
        # Length guard and dedup on canonical candidates
        canon_candidates = [c[:MAX_SQL_LEN] for c in canon_candidates]
        canon_candidates = _dedup_preserve_order(canon_candidates)
        
        # Filter out truncated/broken SQL from canon
        canon_candidates = [c for c in canon_candidates if _is_viable_sql(c)]
        canon_candidates = [c if _is_viable_sql(c) else _clean_bad_tail(c) for c in canon_candidates]
        
        # Schema validation on canonical candidates
        canon_candidates = [c for c in canon_candidates if _uses_only_schema(c, schema_tables, schema_cols)]
        
        # ---- Re-rank by ensuring FROM tables from plan exist in SQL ----
        canon_candidates = _filter_by_plan_tables(canon_candidates, checked)
        candidates       = _filter_by_plan_tables(candidates, checked)
        
        # Score and sort by plan-fit
        canon_candidates = sorted(canon_candidates, key=lambda c: _score_against_plan(c, checked), reverse=True)
        candidates       = sorted(candidates,       key=lambda c: _score_against_plan(c, checked), reverse=True)



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
            # Fast repair path for incomplete input
            if "incomplete input" in (last_error or "").lower():
                quick = _clean_bad_tail(failed_sql_for_repair)
                ok_q, err_q = try_exec(ex["db_path"], quick)
                if ok_q:
                    sql = quick
            
            # If fast repair didn't work, do full LLM repair
            if sql is None:
                rep_prompt = build_repair_prompt(schema, q, checked, failed_sql_for_repair, last_error)
                try:
                    fixed_text = _to_text(ask_with_retry_cached(
    rep_prompt, temperature=0.0, max_output_tokens=800, timeout=120
))

                    fixed_sql = extract_sql(fixed_text)
                    # Apply post-process to repair
                    fixed_sql_processed = _enforce_plan(_add_nocase(fixed_sql), checked)
                    ok2, err_msg2 = try_exec(ex["db_path"], fixed_sql_processed)
                    if ok2:
                        sql = fixed_sql_processed
                    else:
                        # Second repair attempt
                        rep_prompt2 = build_repair_prompt(schema, q, checked, fixed_sql_processed, err_msg2)
                        fixed_text2 = _to_text(ask_with_retry_cached(
    rep_prompt2, temperature=0.0, max_output_tokens=800, timeout=120
))

                        fixed_sql2 = extract_sql(fixed_text2)
                        # Apply post-process to 2nd repair
                        fixed_sql2_processed = _enforce_plan(_add_nocase(fixed_sql2), checked)
                        ok3, _ = try_exec(ex["db_path"], fixed_sql2_processed) # Don't care about 3rd error
                        sql = fixed_sql2_processed if ok3 else "SELECT 1;"
                except Exception:
                    sql = "SELECT 1;"
        
        # Fallback if sql is still None
        if sql is None:
            sql = "SELECT 1;"

        # UPDATE J: Debug logging
        try:
            with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as dbg:
                db_id = ex.get('db_id', 'unknown_db')
                dbg.write(f"\n--- IDX {i} ({db_id}) ---\nQ: {q}\nPLAN:\n{plan}\nCHECKED:\n{checked}\nRAW_SQL_TEXTS:\n{raw_sql_texts}\nCANDS:\n{candidates}\nRAW_CANON_TEXTS:\n{canon_raw_texts}\nCANON_CANDS:\n{canon_candidates}\nLAST_ERROR: {last_error}\nFINAL_SQL:\n{sql}\n\n")
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


if __name__ == "__main__":
    run()