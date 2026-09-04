"""Phase 3 step 5 — sweep BM25 k1/b on the dev set.

    python -m src.retrieval.tune --index indexes/bm25-base

The index is built ONCE; each config only costs search time, so a 20-config
sweep is far cheaper than it sounds.

Selects by `eligible/ndcg_cut_10`, NOT `official/ndcg_cut_10`. The official
scale scores EXCLUDED trials positively, so optimizing for it means
optimizing for finding trials the patient CANNOT enter — the opposite of
this project's goal. See src/eval/metrics.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.eval import data, metrics
from src.retrieval.bm25 import search

# Initial grid [0.6..1.8] x [0.3..0.9] converged at the EDGE (k1=1.8, b=0.9),
# meaning the true optimum was outside it. Widened: b caps at 1.0 by BM25's
# definition; k1 is unbounded, so extended to 4.0.
K1_GRID = [0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 4.0]
B_GRID = [0.5, 0.75, 0.9, 1.0]
SELECT_BY = "eligible/ndcg_cut_10"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="indexes/bm25-base")
    ap.add_argument("--year", type=int, default=data.DEV_YEAR)
    ap.add_argument("--query-mode", default="raw", choices=["raw", "nodeid"])
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="results/_tune_grid.json")
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("Tu choi tune tren tap test 2022.", file=sys.stderr)
        return 1

    topics = data.load_topics(args.year)
    qrels = data.load_qrels(args.year)
    print(f"Quet {len(K1_GRID)*len(B_GRID)} cau hinh tren {len(topics)} topic dev\n")
    print(f"{'k1':>5s} {'b':>5s} {'elig nDCG@10':>13s} {'chinh thuc':>11s} "
          f"{'contam@10':>10s} {'judged@10':>10s}")
    print("-" * 60)

    rows = []
    t0 = time.time()
    for k1 in K1_GRID:
        for b in B_GRID:
            run = search(args.index, topics, k1, b, args.depth,
                         args.query_mode, args.threads)
            agg = metrics.aggregate(metrics.evaluate(run, qrels))
            rows.append({"k1": k1, "b": b, **agg})
            print(f"{k1:5.1f} {b:5.2f} {agg[SELECT_BY]:13.4f} "
                  f"{agg['official/ndcg_cut_10']:11.4f} "
                  f"{agg['elig/contamination_10']:10.4f} "
                  f"{agg['elig/judged_10']:10.4f}", flush=True)

    best = max(rows, key=lambda r: r[SELECT_BY])
    print("-" * 60)
    print(f"\nTot nhat theo {SELECT_BY}:  k1={best['k1']}  b={best['b']}  "
          f"= {best[SELECT_BY]:.4f}")
    best_off = max(rows, key=lambda r: r["official/ndcg_cut_10"])
    if (best_off["k1"], best_off["b"]) != (best["k1"], best["b"]):
        print(f"  (theo thang chinh thuc lai la k1={best_off['k1']} b={best_off['b']} "
              f"— hai thang chon khac nhau, ghi vao bao cao)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"select_by": SELECT_BY, "best": best, "grid": rows}, fh, indent=2)
    print(f"\nLuoi day du: {args.out}   ({time.time()-t0:.0f}s)")
    print(f"\nSinh run tot nhat:\n  PYTHONPATH=. .venv/bin/python -m src.retrieval.bm25 "
          f"--index {args.index} --k1 {best['k1']} --b {best['b']} "
          f"--out runs/bm25_tuned.dev.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
