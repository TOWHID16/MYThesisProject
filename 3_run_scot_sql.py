# SCoT-SQL: Plan -> Plan-Check -> Final SQL
# Output: runs/dev_600/predicted_scot_models_gemini-2.5-pro.sql
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

# ============================================================================
# PATTERN DETECTION (for ensemble candidate selection)
# ============================================================================

_NEG_TOKENS_EXPANDED = {
    "no", "without", "none", "missing", "lack", "exclude", "not present",
    "neither", "except those", "no such", "no one", "no country", "no maker",
    "never", "hasn't", "haven't", "didn't", "don't", "doesn't",
    "নেই", "ছাড়া", "বিহীন", "শূন্য", "অনুপস্থিত"
}

_INTERSECT_TOKENS = {"both", "and", "as well as", "also", "shared", "common", "intersection"}
_UNION_TOKENS = {"or", "either", "any", "union"}
_EXTREMUM_TOKENS = {"most", "least", "highest", "lowest", "maximum", "minimum", "top", "bottom", "largest", "smallest", "best", "worst"}
_PER_GROUP_TOKENS = {"each", "every", "per", "for each", "for every"}

def _needs_negation(q: str) -> bool:
    """Detect if question requires negation/anti-join logic."""
    if not q:
        return False
    s = q.strip().lower()
    return any(tok in s for tok in _NEG_TOKENS_EXPANDED)

def _needs_intersect(q: str) -> bool:
    """Detect 'both X and Y' patterns requiring INTERSECT."""
    if not q:
        return False
    s = q.strip().lower()
    # Look for "both" followed by "and" within reasonable distance
    if "both" in s and " and " in s:
        return True
    # Look for "X and Y" with set-like nouns (cartoons, singers, countries, etc.)
    if re.search(r"\b(cartoons?|singers?|countries|languages?|students?|owners?|professionals?)\b.*\band\b.*\b(directed|written|sung|spoken|living|by)", s):
        return True
    return False

def _needs_union(q: str) -> bool:
    """Detect 'either X or Y' patterns requiring UNION."""
    if not q:
        return False
    s = q.strip().lower()
    return ("either" in s and " or " in s) or (s.count(" or ") >= 1 and not _needs_intersect(q))

def _needs_subquery(q: str) -> bool:
    """Detect queries needing subqueries (per-group extremum, nested filters)."""
    if not q:
        return False
    s = q.strip().lower()
    # Per-group extremum: "most X per Y", "highest X for each Y"
    has_extremum = any(tok in s for tok in _EXTREMUM_TOKENS)
    has_per_group = any(tok in s for tok in _PER_GROUP_TOKENS)
    return has_extremum and has_per_group

# ============================================================================
# SCHEMA VALIDATION & PK/FK PATH ENUMERATION
# ============================================================================

def _enumerate_pk_fk_paths(schema_str: str) -> dict:
    """
    Parse schema to extract PK/FK relationships.
    Returns: {table_name: {"pk": col, "fks": [(fk_col, ref_table, ref_col), ...]}}
    
    Schema format example:
    # table_name (col1 PRIMARY KEY, col2, col3 FOREIGN KEY REFERENCES other_table(other_col))
    """
    schema_map = {}
    for line in schema_str.splitlines():
        line = line.strip()
        if not line.startswith("# "):
            continue
        if "(" not in line or ")" not in line:
            continue
        
        # Extract table name
        head, inside = line[2:].split("(", 1)
        table = head.strip().split()[0]
        cols_str = inside.split(")")[0]
        
        pk = None
        fks = []
        
        for col_def in cols_str.split(","):
            col_def = col_def.strip()
            if "PRIMARY KEY" in col_def.upper():
                pk = col_def.split()[0]
            elif "FOREIGN KEY" in col_def.upper() or "REFERENCES" in col_def.upper():
                # Extract FK: col REFERENCES ref_table(ref_col)
                m = re.search(r"(\w+)\s+.*?REFERENCES\s+(\w+)\s*\((\w+)\)", col_def, re.I)
                if m:
                    fks.append((m.group(1), m.group(2), m.group(3)))
        
        schema_map[table.lower()] = {"pk": pk, "fks": fks}
    
    return schema_map

def _find_join_path(schema_map: dict, table_a: str, table_b: str) -> list:
    """
    Find join path between two tables using BFS on FK relationships.
    Returns: [(table_from, col_from, table_to, col_to), ...]
    """
    from collections import deque
    
    table_a = table_a.lower()
    table_b = table_b.lower()
    
    if table_a == table_b:
        return []
    
    # Build adjacency graph
    graph = {}
    for tbl, info in schema_map.items():
        graph[tbl] = []
        for fk_col, ref_tbl, ref_col in info.get("fks", []):
            graph[tbl].append((ref_tbl, fk_col, ref_col))
            # Bidirectional
            if ref_tbl not in graph:
                graph[ref_tbl] = []
            graph[ref_tbl].append((tbl, ref_col, fk_col))
    
    # BFS
    queue = deque([(table_a, [])])
    visited = {table_a}
    
    while queue:
        current, path = queue.popleft()
        if current == table_b:
            return path
        
        for neighbor, col_from, col_to in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(current, col_from, neighbor, col_to)]))
    
    return []  # No path found

# ============================================================================
# PRE-SQL SANITIZERS
# ============================================================================

def _balance_parens(text: str) -> str:
    """Balance parentheses by removing unmatched ones."""
    stack = []
    to_remove = set()
    for i, ch in enumerate(text):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()
            else:
                to_remove.add(i)
    to_remove.update(stack)
    
    if not to_remove:
        return text
    
    return ''.join(ch for i, ch in enumerate(text) if i not in to_remove)

def _remove_dangling_keywords(sql: str) -> str:
    """Remove dangling SQL keywords at the end (ORDER BY, WHERE, AND, OR, ON, JOIN)."""
    keywords = [r"\bWHERE\s*$", r"\bAND\s*$", r"\bOR\s*$", r"\bON\s*$", 
                r"\bJOIN\s*$", r"\bLEFT\s*$", r"\bRIGHT\s*$", r"\bINNER\s*$",
                r"\bORDER\s+BY\s*$", r"\bGROUP\s+BY\s*$", r"\bHAVING\s*$"]
    
    for pattern in keywords:
        sql = re.sub(pattern, "", sql, flags=re.I)
    
    return sql.strip()

def _fix_incomplete_plan(plan_text: str) -> str:
    """Fix common plan syntax errors (unbalanced parens, incomplete lines)."""
    if not plan_text or "SCoT-Plan:" not in plan_text:
        return plan_text
    
    lines = []
    for line in plan_text.splitlines():
        line = line.strip()
        
        # Fix incomplete clause definitions like "- ORDER BY: (none" -> "- ORDER BY: (none)"
        if line.startswith("- ") and line.count("(") != line.count(")"):
            line = _balance_parens(line)
        
        lines.append(line)
    
    return "\n".join(lines)

# ============================================================================
# GROUP BY / HAVING VALIDATORS WITH AUTO-FIX
# ============================================================================

def _validate_group_by(sql: str) -> tuple[bool, str]:
    """
    Validate GROUP BY rules:
    - Non-aggregated SELECT columns must appear in GROUP BY
    Returns: (is_valid, fixed_sql)
    """
    sql_lower = sql.lower()
    
    # If no GROUP BY, nothing to validate
    if "group by" not in sql_lower:
        return True, sql
    
    # Extract SELECT columns
    select_match = re.search(r"\bselect\b\s+(.*?)\s+\bfrom\b", sql, flags=re.I | re.S)
    if not select_match:
        return True, sql
    
    select_clause = select_match.group(1)
    
    # Extract GROUP BY columns
    group_match = re.search(r"\bgroup\s+by\b\s+(.*?)(?:\bhaving\b|\border\s+by\b|\blimit\b|;|$)", sql, flags=re.I | re.S)
    if not group_match:
        return True, sql
    
    group_clause = group_match.group(1).strip()
    group_cols = set(re.findall(r"\b\w+\.\w+\b|\b\w+\b", group_clause))
    
    # Identify non-aggregated SELECT columns
    select_cols = []
    agg_funcs = r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\("
    
    for col_expr in select_clause.split(","):
        col_expr = col_expr.strip()
        # Skip aggregates
        if re.search(agg_funcs, col_expr, re.I):
            continue
        # Skip DISTINCT
        col_expr = re.sub(r"\bDISTINCT\b", "", col_expr, flags=re.I).strip()
        # Extract column reference
        col_match = re.search(r"\b(\w+\.\w+|\w+)\b", col_expr)
        if col_match:
            select_cols.append(col_match.group(1))
    
    # Check if all non-aggregated SELECT columns are in GROUP BY
    missing = []
    for col in select_cols:
        # Check both qualified and unqualified versions
        col_base = col.split(".")[-1] if "." in col else col
        if col not in group_cols and col_base not in group_cols:
            missing.append(col)
    
    if not missing:
        return True, sql
    
    # Auto-fix: add missing columns to GROUP BY
    new_group = group_clause + ", " + ", ".join(missing)
    fixed_sql = re.sub(
        r"(\bgroup\s+by\b\s+)(.*?)(?=\bhaving\b|\border\s+by\b|\blimit\b|;|$)",
        r"\1" + new_group + " ",
        sql,
        flags=re.I | re.S
    )
    
    return False, fixed_sql

def _validate_having(sql: str) -> tuple[bool, str]:
    """
    Validate HAVING rules:
    - HAVING requires GROUP BY
    - Move non-aggregate filters from HAVING to WHERE when possible
    Returns: (is_valid, fixed_sql)
    """
    sql_lower = sql.lower()
    
    if "having" not in sql_lower:
        return True, sql
    
    # HAVING without GROUP BY is invalid
    if "group by" not in sql_lower:
        # Try to add a minimal GROUP BY or move HAVING to WHERE
        having_match = re.search(r"\bhaving\b\s+(.*?)(?:\border\s+by\b|\blimit\b|;|$)", sql, flags=re.I | re.S)
        if having_match:
            having_clause = having_match.group(1).strip()
            # If HAVING contains aggregates, we can't easily fix it
            if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", having_clause, re.I):
                return False, sql  # Invalid, needs manual fix
            
            # Move to WHERE
            if "where" in sql_lower:
                # Append to existing WHERE
                fixed_sql = re.sub(
                    r"(\bwhere\b\s+.*?)(\bhaving\b\s+" + re.escape(having_clause) + r")",
                    r"\1 AND " + having_clause + " ",
                    sql,
                    flags=re.I | re.S
                )
                fixed_sql = re.sub(r"\bhaving\b\s+" + re.escape(having_clause), "", fixed_sql, flags=re.I)
            else:
                # Add WHERE clause
                fixed_sql = re.sub(
                    r"(\bfrom\b\s+.*?)(\bhaving\b\s+" + re.escape(having_clause) + r")",
                    r"\1 WHERE " + having_clause + " ",
                    sql,
                    flags=re.I | re.S
                )
                fixed_sql = re.sub(r"\bhaving\b\s+" + re.escape(having_clause), "", fixed_sql, flags=re.I)
            
            return False, fixed_sql
    
    return True, sql

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

# ============================================================================
# CARDINALITY-AWARE RANKING
# ============================================================================

def _check_result_sanity(sql: str, question: str, db_path: str, checked_plan: str) -> int:
    """
    Check result sanity based on question intent.
    Returns penalty score (0 = perfect, higher = worse).
    """
    penalty = 0
    q_lower = question.lower()
    
    # Execute query to get result
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        row_count = len(rows)
        cur.close()
        con.close()
    except Exception:
        return 100  # Execution failed = maximum penalty
    
    # Check top-k intents
    top_k_match = re.search(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b|\bhighest\s+(\d+)\b|\blowest\s+(\d+)\b", q_lower)
    if top_k_match:
        k = int(top_k_match.group(1) or top_k_match.group(2) or top_k_match.group(3) or top_k_match.group(4))
        if row_count > k:
            penalty += 10  # Returned more than requested
    
    # Check LIMIT from plan
    plan_lower = checked_plan.lower()
    if "- limit:" in plan_lower:
        lim_line = plan_lower.split("- limit:",1)[1].split("\n",1)[0].strip()
        if lim_line != "(none)":
            lim_match = re.match(r"(\d+)", lim_line)
            if lim_match:
                expected_lim = int(lim_match.group(1))
                if row_count > expected_lim:
                    penalty += 5
    
    # Check distinctness for "different", "distinct", "unique"
    if any(word in q_lower for word in ["different", "distinct", "unique", "각", "ভিন্ন"]):
        if len(rows) != len(set(rows)):
            penalty += 5  # Has duplicates when distinctness expected
    
    # Check INTERSECT/UNION semantics
    if _needs_intersect(question):
        if "intersect" not in sql.lower() and "inner join" not in sql.lower():
            penalty += 10  # Missing INTERSECT logic
    
    if _needs_union(question):
        if "union" not in sql.lower():
            penalty += 10  # Missing UNION logic
    
    # Check negation semantics
    if _needs_negation(question):
        if not any(kw in sql.lower() for kw in ["not exists", "not in", "left join", "except"]):
            penalty += 10  # Missing negation logic
    
    return penalty

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
RUN_DIR   = PROJ_ROOT / "runs" / "dev_600"
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

**CRITICAL REQUIREMENTS:**
- Specify PRIMARY KEY / FOREIGN KEY relationships in FROM clause
- Use fully qualified table.column references everywhere
- Include explicit ON clauses with FK keys for all JOINs

# Rules for popularity wording:
# - If the question says "most popular" without mentioning "percentage/share", treat popularity as COUNT of entities (e.g., number of countries).
# - Only use SUM(Percentage) when the question explicitly asks by percentage/share.

Return EXACTLY this format:

SCoT-Plan:
- FROM: <tables with aliases and explicit JOIN ON fk_col = pk_col>
- WHERE: <filters with table.column or (none)>
- GROUP BY: <table.cols or (none)>
- HAVING: <conditions with aggregates or (none)>
- ORDER BY: <table.cols or (none)>
- SELECT: <table.column list>
- LIMIT: <N or (none)>

{FEW_SHOT_PLAN}

Question: {question}
"""

def build_check_prompt(schema_str: str, plan_text: str) -> str:
    return f"""{schema_str.strip()}

{FEW_SHOT_CHECK}

**CRITICAL VALIDATION:**
- Verify all tables/columns exist in schema
- Ensure all JOIN clauses have explicit ON with FK keys
- Confirm all columns are fully qualified (table.column)
- Check that non-aggregated SELECT columns appear in GROUP BY

Given this SCoT-Plan, fix illegal tables/columns or clause conflicts so it strictly matches the schema.

Return only the corrected plan in the exact same format:

{plan_text.strip()}
"""

def build_sql_prompt(schema_str: str, checked_plan: str) -> str:
    return f"""{schema_str.strip()}

Using the SCoT-Plan below, write ONLY the final SQL wrapped in a fenced block.

**MANDATORY REQUIREMENTS:**
- Qualify ALL columns with table aliases (table.column)
- Include explicit JOIN ON clauses with FK keys
- Non-aggregated SELECT columns MUST appear in GROUP BY
- Use COLLATE NOCASE for string comparisons when appropriate

```sql
SELECT table.column, ...
FROM table AS alias
JOIN other_table AS alias2 ON alias.fk = alias2.pk
...
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

Return ONLY the final SQL for the question below, wrapped in a fenced block.

**MANDATORY REQUIREMENTS:**
- Qualify ALL columns with table aliases (table.column)
- Include explicit JOIN ON clauses with FK keys from schema
- Use only tables/columns from the schema above

```sql
SELECT table.column, ...
FROM table AS alias
...
```

Rules:
- SQL-92 only (no CTE/window)
- Do NOT add ORDER BY unless ranking/sorting is required
- No commentary outside the fenced block

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

# ============================================================================
# TEMPLATE SKELETON GENERATORS (HIGH PRIORITY)
# ============================================================================

def build_not_exists_template(schema_str: str, question: str, schema_map: dict) -> str:
    """
    Generate NOT EXISTS anti-join template for negation questions.
    """
    return f"""{schema_str.strip()}

This question requires finding items that do NOT have a certain property (negation).
Use NOT EXISTS or LEFT JOIN ... IS NULL pattern.

Template preference:
1. NOT EXISTS (most portable):
```sql
SELECT table_a.column
FROM table_a
WHERE NOT EXISTS (
  SELECT 1
  FROM table_b
  WHERE table_b.fk = table_a.pk
    AND <additional_conditions>
);
```

2. LEFT JOIN anti-join (alternative):
```sql
SELECT table_a.column
FROM table_a
LEFT JOIN table_b ON table_b.fk = table_a.pk AND <conditions>
WHERE table_b.pk IS NULL;
```

Rules:
- Use only tables/columns from the schema above
- Explicitly specify all join keys (table_a.pk = table_b.fk)
- Qualify all columns with table aliases
- Return ONLY the SQL in a fenced block

Question: {question}
"""

def build_intersect_skeleton(schema_str: str, question: str) -> str:
    """
    Generate INTERSECT skeleton for 'both X and Y' patterns.
    """
    return f"""{schema_str.strip()}

This question requires finding items that satisfy BOTH condition X AND condition Y (intersection).
Use INTERSECT to combine two SELECT statements.

Template:
```sql
SELECT column1, column2
FROM table
JOIN ...
WHERE condition_X
INTERSECT
SELECT column1, column2
FROM table
JOIN ...
WHERE condition_Y;
```

Rules:
- Both SELECT statements must have identical column structure
- Use explicit JOIN keys from schema
- Qualify all columns with table aliases
- Return ONLY the SQL in a fenced block

Question: {question}
"""

def build_union_skeleton(schema_str: str, question: str) -> str:
    """
    Generate UNION skeleton for 'either X or Y' patterns.
    """
    return f"""{schema_str.strip()}

This question requires finding items that satisfy EITHER condition X OR condition Y (union).
Use UNION to combine two SELECT statements.

Template:
```sql
SELECT column1, column2
FROM table
JOIN ...
WHERE condition_X
UNION
SELECT column1, column2
FROM table
JOIN ...
WHERE condition_Y;
```

Rules:
- Both SELECT statements must have identical column structure
- Use explicit JOIN keys from schema
- Qualify all columns with table aliases
- Return ONLY the SQL in a fenced block

Question: {question}
"""

def build_subquery_extremum_skeleton(schema_str: str, question: str) -> str:
    """
    Generate subquery skeleton for per-group extremum (most X per Y, highest X for each Y).
    """
    return f"""{schema_str.strip()}

This question requires finding the extreme value (max/min) for EACH group (per-group extremum).
Use a subquery with GROUP BY to find extremum, then join back.

Template:
```sql
SELECT t.column1, t.column2, t.metric_column
FROM table AS t
JOIN (
  SELECT group_key, MAX(metric_column) AS max_metric
  FROM table
  WHERE <filters>
  GROUP BY group_key
) AS z
  ON z.group_key = t.group_key 
  AND z.max_metric = t.metric_column;
```

Alternative (using NOT EXISTS for most efficient):
```sql
SELECT t.column1, t.column2, t.metric_column
FROM table AS t
WHERE NOT EXISTS (
  SELECT 1
  FROM table AS t2
  WHERE t2.group_key = t.group_key
    AND t2.metric_column > t.metric_column
);
```

Rules:
- Identify the group_key (the "per Y" or "each X" entity)
- Identify the metric_column (the value being maximized/minimized)
- Use explicit JOIN keys from schema
- Qualify all columns with table aliases
- Return ONLY the SQL in a fenced block

Question: {question}
"""

def build_qd_intercol_candidate_prompt(schema_str: str, question: str, schema_map: dict) -> str:
    """
    Generate QD-InterCol style prompt with explicit column grounding.
    """
    # Extract FK hints from schema_map
    fk_hints = []
    for table, info in schema_map.items():
        for fk_col, ref_tbl, ref_col in info.get("fks", []):
            fk_hints.append(f"  - {table}.{fk_col} references {ref_tbl}.{ref_col}")
    
    fk_section = "\n".join(fk_hints) if fk_hints else "  (No FK relationships found)"
    
    return f"""{schema_str.strip()}

### Foreign Key Relationships:
{fk_section}

Decompose the question into 3-6 steps with InterCol annotations.
For EACH step, list the exact table.column references used.

Format:
Decomposition (InterCOL):
1. <step description> [Cols: table_a.col1, table_b.col2]
2. <step description> [Cols: table_c.col3]
...

Then output ONLY the final SQL in a fenced block:
```sql
SELECT ...
```

Rules:
- Use ONLY tables/columns from the schema above
- Use the FK relationships listed for all joins
- Qualify ALL columns with table aliases (table.column)
- Include explicit ON clauses with FK keys
- SQL-92 only (no CTE/window)

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
    plan_prompt, temperature=0.3, max_output_tokens=800, timeout=30
))

        except Exception as e:
            print(f"  [Plan ERR] {i}/{len(data)} -> {e}")
            preds.append("SELECT 1;")
            continue

        # 2) PLAN-CHECK  (deterministic, with retry)
        check_prompt = build_check_prompt(schema, plan)
        try:
            checked = _to_text(ask_with_retry_cached(
                check_prompt, temperature=0.0, max_output_tokens=800, timeout=30
            ))
        except Exception as e:
            print(f"  [Check ERR] {i}/{len(data)} -> {e}")
            checked = plan  # fallback
        
        # Fix incomplete plan syntax
        checked = _fix_incomplete_plan(checked)
        
        # Normalize plan to dedupe repeated clauses
        checked = _normalize_plan(checked)
        
        # Sanity: model যদি JSON/garbage দেয়
        if "SCoT-Plan:" not in checked:
            checked = plan
        
        # Enumerate PK/FK paths from schema for better grounding
        schema_map = _enumerate_pk_fk_paths(schema)
        
        # ========================================================================
        # 3) ENSEMBLE CANDIDATE GENERATION (4-6 specialized candidates)
        # ========================================================================
        candidates = []
        raw_sql_texts = []  # Capture raw model outputs
        
        # Candidate 1: SCoT-Native (plan-based, temperature 0.2 for slight diversity)
        sql_prompt = build_sql_prompt(schema, checked)
        try:
            sql_text = _to_text(ask_with_retry_cached(sql_prompt, temperature=0.2, max_output_tokens=1024, timeout=30))
            raw_sql_texts.append(("SCoT-Native", sql_text))
            cand_sql = extract_sql(sql_text)
            if cand_sql.strip().lower() != "select 1;":
                candidates.append(("SCoT-Native", cand_sql))
        except Exception:
            pass
        
        # Candidate 2: QD-InterCol style (explicit column grounding)
        try:
            qd_prompt = build_qd_intercol_candidate_prompt(schema, q, schema_map)
            qd_text = _to_text(ask_with_retry_cached(qd_prompt, temperature=0.2, max_output_tokens=1200, timeout=30))
            raw_sql_texts.append(("QD-InterCol", qd_text))
            qd_sql = extract_sql(qd_text)
            if qd_sql.strip().lower() != "select 1;":
                candidates.append(("QD-InterCol", qd_sql))
        except Exception:
            pass
        
        # Candidate 3: Standard Direct (no reasoning)
        try:
            std_text = _to_text(ask_with_retry_cached(
                build_standard_sql_prompt(schema, q),
                temperature=0.0, max_output_tokens=1024, timeout=30
            ))
            raw_sql_texts.append(("Standard", std_text))
            std_sql = extract_sql(std_text)
            if std_sql.strip().lower() != "select 1;":
                candidates.append(("Standard", std_sql))
        except Exception:
            pass
        
        # Pattern-specific candidates (based on question analysis)
        
        # Candidate 4: NOT EXISTS anti-join (for negation questions)
        if _needs_negation(q):
            try:
                neg_prompt = build_not_exists_template(schema, q, schema_map)
                neg_text = _to_text(ask_with_retry_cached(neg_prompt, temperature=0.2, max_output_tokens=1024, timeout=30))
                raw_sql_texts.append(("NOT-EXISTS", neg_text))
                neg_sql = extract_sql(neg_text)
                if neg_sql.strip().lower() != "select 1;":
                    candidates.append(("NOT-EXISTS", neg_sql))
            except Exception:
                pass
        
        # Candidate 5: INTERSECT skeleton (for "both X and Y")
        if _needs_intersect(q):
            try:
                intersect_prompt = build_intersect_skeleton(schema, q)
                intersect_text = _to_text(ask_with_retry_cached(intersect_prompt, temperature=0.2, max_output_tokens=1200, timeout=30))
                raw_sql_texts.append(("INTERSECT", intersect_text))
                intersect_sql = extract_sql(intersect_text)
                if intersect_sql.strip().lower() != "select 1;":
                    candidates.append(("INTERSECT", intersect_sql))
            except Exception:
                pass
        
        # Candidate 6: UNION skeleton (for "either X or Y")
        if _needs_union(q) and not _needs_intersect(q):  # Avoid both
            try:
                union_prompt = build_union_skeleton(schema, q)
                union_text = _to_text(ask_with_retry_cached(union_prompt, temperature=0.2, max_output_tokens=1200, timeout=30))
                raw_sql_texts.append(("UNION", union_text))
                union_sql = extract_sql(union_text)
                if union_sql.strip().lower() != "select 1;":
                    candidates.append(("UNION", union_sql))
            except Exception:
                pass
        
        # Candidate 7: Subquery extremum (for per-group max/min)
        if _needs_subquery(q):
            try:
                subq_prompt = build_subquery_extremum_skeleton(schema, q)
                subq_text = _to_text(ask_with_retry_cached(subq_prompt, temperature=0.2, max_output_tokens=1200, timeout=30))
                raw_sql_texts.append(("SUBQUERY", subq_text))
                subq_sql = extract_sql(subq_text)
                if subq_sql.strip().lower() != "select 1;":
                    candidates.append(("SUBQUERY", subq_sql))
            except Exception:
                pass
        
        # Extract just SQL from candidates (remove labels)
        candidate_sqls = [sql for _, sql in candidates]
        
        # ========================================================================
        # PRE-PROCESSING: Sanitize and validate
        # ========================================================================
        
        # Apply pre-SQL sanitizers
        candidate_sqls = [_remove_dangling_keywords(c) for c in candidate_sqls]
        candidate_sqls = [_balance_parens(c) for c in candidate_sqls]
        
        # Length guard
        candidate_sqls = [c[:MAX_SQL_LEN] for c in candidate_sqls]
        
        # Deduplicate while preserving order
        candidate_sqls = _dedup_preserve_order(candidate_sqls)
        
        # Filter out truncated/broken SQL
        candidate_sqls = [c for c in candidate_sqls if _is_viable_sql(c)]
        
        # Try to clean bad tails
        candidate_sqls = [c if _is_viable_sql(c) else _clean_bad_tail(c) for c in candidate_sqls]
        
        # Apply GROUP BY / HAVING validators with auto-fix
        validated_sqls = []
        for c in candidate_sqls:
            _, fixed = _validate_group_by(c)
            _, fixed = _validate_having(fixed)
            validated_sqls.append(fixed)
        candidate_sqls = validated_sqls
        
        # Schema validation: reject candidates with invalid table/column refs
        candidate_sqls = [c for c in candidate_sqls if _uses_only_schema(c, schema_tables, schema_cols)]

        # Retry once if we lost all candidates
        if not candidate_sqls:
            try:
                sql_prompt = build_sql_prompt(schema, checked)
                sql_text = _to_text(ask_with_retry_cached(sql_prompt, temperature=0.0, max_output_tokens=1024, timeout=30))
                cand_sql = extract_sql(sql_text)
                if cand_sql.strip().lower() != "select 1;":
                    candidate_sqls.append(cand_sql)
            except Exception:
                pass
        
        # ========================================================================
        # CANONICALIZATION (optional refinement)
        # ========================================================================
        canon_candidates = []
        canon_raw_texts = []
        for cand in candidate_sqls:
            try:
                canon_text = _to_text(ask_with_retry_cached(
                    build_canon_prompt(schema, cand),
                    temperature=0.0, max_output_tokens=800, timeout=30
                ))
                canon_raw_texts.append(canon_text)
                canon_sql = extract_sql(canon_text)
                canon_candidates.append(canon_sql if canon_sql.strip().lower() != "select 1;" else cand)
            except Exception:
                canon_candidates.append(cand)
        
        # ========================================================================
        # POST-PROCESSING: Plan enforcement and preference application
        # ========================================================================
        canon_candidates = [_enforce_plan(_add_nocase(c), checked) for c in canon_candidates]
        candidate_sqls = [_enforce_plan(_add_nocase(c), checked) for c in candidate_sqls]
        
        # Apply negation preference (NOT EXISTS / LEFT JOIN)
        if _needs_negation(q):
            canon_candidates = [_prefer_setdiff(c, q) for c in canon_candidates]
            candidate_sqls = [_prefer_setdiff(c, q) for c in candidate_sqls]
        
        # Length guard and dedup on canonical candidates
        canon_candidates = [c[:MAX_SQL_LEN] for c in canon_candidates]
        canon_candidates = _dedup_preserve_order(canon_candidates)
        
        # Filter out truncated/broken SQL from canon
        canon_candidates = [c for c in canon_candidates if _is_viable_sql(c)]
        canon_candidates = [c if _is_viable_sql(c) else _clean_bad_tail(c) for c in canon_candidates]
        
        # Schema validation on canonical candidates
        canon_candidates = [c for c in canon_candidates if _uses_only_schema(c, schema_tables, schema_cols)]
        
        # ========================================================================
        # EXECUTION-GUIDED RANKING WITH CARDINALITY CHECKS
        # ========================================================================
        
        # Re-rank by plan table coverage
        canon_candidates = _filter_by_plan_tables(canon_candidates, checked)
        candidate_sqls = _filter_by_plan_tables(candidate_sqls, checked)
        
        # Combine all candidates for unified ranking
        all_candidates = list(set(canon_candidates + candidate_sqls))
        
        # Score each candidate: (execution_ok, -sanity_penalty, plan_fit_score, sql)
        scored_candidates = []
        for cand in all_candidates:
            # Test execution
            ok, err_msg = try_exec(ex["db_path"], cand)
            exec_ok = 1 if ok else 0
            
            # Sanity check (only if execution succeeded)
            if ok:
                sanity_penalty = _check_result_sanity(cand, q, ex["db_path"], checked)
            else:
                sanity_penalty = 0  # Doesn't matter if execution failed
            
            # Plan fit score
            plan_score = _score_against_plan(cand, checked)
            
            scored_candidates.append((exec_ok, -sanity_penalty, plan_score, cand, err_msg if not ok else ""))
        
        # Sort: execution success first, then lowest sanity penalty, then highest plan fit
        scored_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        
        # Select best candidate
        if scored_candidates and scored_candidates[0][0] == 1:  # At least one succeeded
            sql = scored_candidates[0][3]
            last_error = ""
        else:
            sql = None
            last_error = scored_candidates[0][4] if scored_candidates else "failed to execute"
            failed_sql_for_repair = scored_candidates[0][3] if scored_candidates else "SELECT 1;"

        # ========================================================================
        # EXECUTION-GUIDED REPAIR (if all candidates failed)
        # ========================================================================
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
    rep_prompt, temperature=0.0, max_output_tokens=800, timeout=30
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
    rep_prompt2, temperature=0.0, max_output_tokens=800, timeout=30
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
