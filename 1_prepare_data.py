# 1_prepare_data.py
# Create a small Spider sample set: schema_str + question + gold_query + db_path
# Outputs:
#   runs/dev_30/my_test_set.json
#   runs/dev_30/my_gold_set.sql
#   runs/dev_30/my_gold_set.json

import os, json, random, sys
from pathlib import Path

# ---- CONFIG ----
PROJ_ROOT = Path(__file__).resolve().parent
SPIDER_DIR = PROJ_ROOT / "data" / "spider"   # E:\MYThesisProject\data\spider
OUT_DIR    = PROJ_ROOT / "runs" / "dev_30"   # change to dev_200 later
NUM_SAMPLES = 30                              # sanity first; later 200

# fixed names
DEV_JSON_PATH      = SPIDER_DIR / "dev.json"
TABLES_JSON_PATH   = SPIDER_DIR / "tables.json"
DEV_GOLD_SQL_PATH  = SPIDER_DIR / "dev_gold.sql"

def find_db_root(spider_dir: Path) -> Path:
    cand1 = spider_dir / "database"
    cand2 = spider_dir / "databases"
    if cand1.is_dir(): return cand1
    if cand2.is_dir(): return cand2
    raise FileNotFoundError("Spider database/ or databases/ folder not found")

def format_schema_api_docs(schema_info: dict) -> str:
    lines = ["### SQLite SQL tables, with their properties:", "#"]
    table_names = schema_info["table_names_original"]
    col_pairs   = schema_info["column_names_original"]  # list of [table_idx, col_name]

    # collect columns by table
    table_cols = {i: [] for i in range(len(table_names))}
    for t_idx, c_name in col_pairs:
        # skip special/global "*" row and non-table columns
        if t_idx == -1 or c_name == "*":
            continue
        table_cols[t_idx].append(c_name)

    # render tables
    for i, tname in enumerate(table_names):
        cols = ", ".join(table_cols.get(i, []))
        lines.append(f"# {tname} ({cols})")

    # foreign key hints (more grounding for joins)
    fks = schema_info.get("foreign_keys", [])
    for (from_id, to_id) in fks:
        # from_id / to_id are indices into col_pairs
        ft, fc = col_pairs[from_id]   # (table_idx, col_name)
        tt, tc = col_pairs[to_id]
        if ft != -1 and tt != -1 and fc != "*" and tc != "*":
            from_tbl = table_names[ft]
            to_tbl   = table_names[tt]
            lines.append(f"# FK: {from_tbl}.{fc} -> {to_tbl}.{tc}")

    return "\n".join(lines) + "\n"


def main():
    # ---- load files
    if not DEV_JSON_PATH.exists() or not TABLES_JSON_PATH.exists() or not DEV_GOLD_SQL_PATH.exists():
        print("[ERR] Spider files missing. Check data/spider/ paths.")
        sys.exit(1)

    dev_data   = json.loads(DEV_JSON_PATH.read_text(encoding="utf-8"))
    tables_all = json.loads(TABLES_JSON_PATH.read_text(encoding="utf-8"))

    # CLEAN gold_sql: drop trailing "\t<db_id>"
    gold_sql_list = []
    for ln in DEV_GOLD_SQL_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if "\t" in ln:
            ln = ln.split("\t", 1)[0].strip()
        gold_sql_list.append(ln)

    if len(dev_data) != len(gold_sql_list):
        print(f"[WARN] dev.json ({len(dev_data)}) != dev_gold.sql ({len(gold_sql_list)})")

    # map db_id -> schema
    schemas_dict = {db["db_id"]: db for db in tables_all}

    # db root for sqlite paths
    db_root = find_db_root(SPIDER_DIR)

    all_items = []
    for i, ex in enumerate(dev_data):
        db_id = ex["db_id"]
        q    = ex["question"]
        gold = gold_sql_list[i] if i < len(gold_sql_list) else ""

        schema = schemas_dict.get(db_id)
        if not schema:
            # rare: missing schema entry
            continue

        schema_str = format_schema_api_docs(schema)

        # sqlite path like data/spider/database/{db_id}/{db_id}.sqlite
        db_file = db_root / db_id / f"{db_id}.sqlite"
        if not db_file.exists():
            # some repos name file as database.sqlite; try fallback
            alt = db_root / db_id / "database.sqlite"
            db_path = str(alt) if alt.exists() else str(db_file)
        else:
            db_path = str(db_file)

        all_items.append({
            "db_id": db_id,
            "question": q,
            "schema_str": schema_str,
            "gold_query": gold,
            "db_path": db_path,
        })

    random.seed(42)
    random.shuffle(all_items)
    sample = all_items[:NUM_SAMPLES]

    # ---- write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_test = OUT_DIR / "my_test_set.json"
    out_gold_sql = OUT_DIR / "my_gold_set.sql"
    out_gold_json = OUT_DIR / "my_gold_set.json"

    out_test.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_gold_sql.open("w", encoding="utf-8") as f:
        for e in sample:
            f.write(e["gold_query"].strip() + "\n")
    out_gold_json.write_text(
        json.dumps([{"db_id": e["db_id"], "query": e["gold_query"]} for e in sample],
                   indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[OK] Wrote: {out_test}")
    print(f"[OK] Wrote: {out_gold_sql}")
    print(f"[OK] Wrote: {out_gold_json}")
    print(f"[INFO] Total sampled: {len(sample)} from {len(all_items)}")

if __name__ == "__main__":
    main()
