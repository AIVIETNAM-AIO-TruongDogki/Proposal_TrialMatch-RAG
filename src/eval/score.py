"""Cham diem mot run file. Day la lenh duy nhat de bao cao ket qua.

    python -m src.eval.score runs/bm25.dev.txt --year 2021
    python -m src.eval.score runs/hybrid.dev.txt --year 2021 --vs results/bm25.dev.json
"""

from __future__ import annotations

import argparse
import os
import sys

from src.eval import data, metrics, run_io, sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run file dinh dang TREC")
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--rawdata", default=data.RAWDATA)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None, help="ten ban ghi (mac dinh: ten file run)")
    ap.add_argument("--vs", default=None, help="file results/*.json de so sanh")
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! Dang cham tren TAP TEST 2022. Theo plan, tap nay chi duoc cham\n"
              "!! MOT LAN o Phase 11. Neu ban dang tune tham so, hay dung --year 2021.\n",
              file=sys.stderr)

    qrels = data.load_qrels(args.year, args.rawdata)
    run = run_io.read_run(args.run)

    missing = set(qrels) - set(run)
    if missing:
        print(f"Canh bao: run thieu {len(missing)}/{len(qrels)} topic "
              f"(vd {sorted(missing)[:3]}) — chung se tinh diem 0.", file=sys.stderr)

    per = metrics.evaluate(run, qrels)
    agg = metrics.aggregate(per)
    n = len(qrels)

    tag = args.tag or os.path.splitext(os.path.basename(args.run))[0]
    print(metrics.format_report(agg, n, title=f"{tag}   (qrels {args.year})"))

    path = run_io.log_result(args.results, tag, agg, per,
                             {"run": args.run, "year": args.year})
    print(f"\nDa ghi: {path}")

    if args.vs:
        base = run_io.load_result(args.vs)
        print(sig.compare(per, base["per_topic"],
                          ["official/ndcg_cut_10", "eligible/ndcg_cut_10",
                           "eligible/P_10", "elig/contamination_10",
                           "elig/recall_1000"],
                          name_a=tag, name_b=base["tag"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
