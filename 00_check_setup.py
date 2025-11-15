# 00_check_setup.py
import os, json, glob, sys

# Spider ডিরেক্টরি (আমরা data/spider ধরছি)
SPIDER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'spider'))

def check_spider():
    print(f"[i] Using SPIDER_DIR = {SPIDER_DIR}")
    required = ['dev.json', 'tables.json', 'dev_gold.sql']
    ok = True
    for name in required:
        path = os.path.join(SPIDER_DIR, name)
        exists = os.path.isfile(path)
        print(f" - {name:12s}: {'OK' if exists else 'MISSING'}  -> {path}")
        ok &= exists

    # database/ অথবা databases/
    db_root = os.path.join(SPIDER_DIR, 'database')
    if not os.path.isdir(db_root):
        alt = os.path.join(SPIDER_DIR, 'databases')
        db_root = alt if os.path.isdir(alt) else db_root
    print(f" - database dir : {'OK' if os.path.isdir(db_root) else 'MISSING'} -> {db_root}")

    if ok and os.path.isdir(db_root):
        # JSON লোড করে একটু স্যানিটি-চেক
        dev = json.load(open(os.path.join(SPIDER_DIR, 'dev.json'), 'r', encoding='utf-8'))
        tables = json.load(open(os.path.join(SPIDER_DIR, 'tables.json'), 'r', encoding='utf-8'))
        print(f"[i] dev.json questions = {len(dev)} ; tables entries = {len(tables)}")

        # প্রথম উদাহরণের db ফাইল আছে কি না
        db_id = dev[0]['db_id']
        sqlite_path = None
        for root in ('database', 'databases'):
            cand = os.path.join(SPIDER_DIR, root, db_id, f"{db_id}.sqlite")
            if os.path.isfile(cand):
                sqlite_path = cand; break
        print(f" - sample sqlite : {'OK' if sqlite_path else 'MISSING'} -> {sqlite_path}")

    return ok and os.path.isdir(db_root)

if __name__ == "__main__":
    ok = check_spider()
    print("\n[RESULT] Setup", "OK ✅" if ok else "NOT READY ❌")
    sys.exit(0 if ok else 1)
