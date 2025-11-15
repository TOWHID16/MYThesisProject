# 5_diff_methods.py
import csv, sys
from pathlib import Path

def read_csv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def index_by_idx(rows):
    return {int(r["idx"]): r for r in rows}

def main():
    if len(sys.argv) != 3:
        print("Usage: python 5_diff_methods.py <csv_A> <csv_B>")
        sys.exit(1)
    A = read_csv(sys.argv[1])
    B = read_csv(sys.argv[2])
    ia = index_by_idx(A); ib = index_by_idx(B)

    wins = []; losses = []
    for k in ia:
        ra = ia[k]; rb = ib.get(k)
        if not rb: continue
        ma = ra["match"] == "True"
        mb = rb["match"] == "True"
        if ma and not mb:
            wins.append(k)
        if mb and not ma:
            losses.append(k)

    print(f"A wins over B on: {wins}")
    print(f"B wins over A on: {losses}")

if __name__ == "__main__":
    main()
