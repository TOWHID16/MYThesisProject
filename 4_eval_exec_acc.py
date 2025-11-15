# 4_eval_exec_acc.py
# Quick execution-accuracy checker for Spider (dev_30)
# Evaluates multiple prediction files; writes one CSV per file + a summary.

import sqlite3, json, csv
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
RUN_DIR    = PROJ_ROOT / "runs" / "dev_30"
TEST_JSON  = RUN_DIR / "my_test_set.json"

# ======= Add as many pred files as you want =======
PRED_FILES = [
    RUN_DIR / "predicted_scot_models_gemini-2.5-pro.sql",
    RUN_DIR / "predicted_cot_models_gemini-2.5-pro.sql",
    RUN_DIR / "predicted_standard.sql",
    RUN_DIR / "predicted_qdecomp.sql",
    RUN_DIR / "predicted_qd_intercol.sql",
]
# ===================================================

def norm_cell(x):
    try:
        if x is None: return ""
        s = str(x).strip()
        # keep numbers' casing; lower for text
        try:
            float(s)
            return s
        except:
            return s.lower()
    except Exception:
        return str(x)

def run_query(db_path, sql):
    """returns (ok, rows_or_err) ; rows normalized+sorted"""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close(); con.close()
        norm = [tuple(norm_cell(c) for c in r) for r in rows]
        norm.sort()
        return True, norm
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def eval_one(pred_file: Path):
    data  = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    preds = [ln.strip() for ln in pred_file.read_text(encoding="utf-8").splitlines()]
    assert len(preds) == len(data), f"[{pred_file.name}] Pred lines {len(preds)} != test size {len(data)}"

    total = len(data)
    correct = 0
    rows = []

    for i, (ex, pred_sql) in enumerate(zip(data, preds), 1):
        db = ex["db_path"]
        gold_sql = ex["gold_query"].strip().rstrip(";") + ";"
        pred_sql = pred_sql.strip().rstrip(";") + ";"

        ok_g, gold_res = run_query(db, gold_sql)
        ok_p, pred_res = run_query(db, pred_sql)

        match = ok_g and ok_p and (gold_res == pred_res)
        if match:
            correct += 1

        rows.append({
            "idx": i,
            "db_id": ex["db_id"],
            "question": ex["question"],
            "pred_ok": ok_p,
            "gold_ok": ok_g,
            "match": match,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "pred_err_or_rows": pred_res,
            "gold_rows": gold_res,
        })

        if i % 10 == 0:
            print(f"[{pred_file.name}] evaluated {i}/{total}")

    acc = correct / total * 100.0
    print(f"\n[{pred_file.name}] Execution accuracy: {correct}/{total} = {acc:.1f}%")

    out_csv = RUN_DIR / f"exec_eval_report__{pred_file.stem}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] Wrote per-item report -> {out_csv}")

    return {"pred_file": pred_file.name, "correct": correct, "total": total, "acc": acc, "csv": out_csv.name}

def main():
    summary = []
    for pf in PRED_FILES:
        print(f"\n=== Evaluating: {pf.name} ===")
        summary.append(eval_one(pf))

    # write a small summary
    sum_csv = RUN_DIR / "exec_eval_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pred_file","correct","total","acc","csv"])
        w.writeheader(); w.writerows(summary)
    print(f"\n[OK] Wrote summary -> {sum_csv}")

if __name__ == "__main__":
    main()
