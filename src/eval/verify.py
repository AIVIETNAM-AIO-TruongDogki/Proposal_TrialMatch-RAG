"""Kiem chung harness — exit criterion cua Phase 2.

    python -m src.eval.verify

Ba phep kiem:
  1. nDCG@10 khop voi ban tinh tay (harness co noi dung day khong?)
  2. contamination@10 khop voi ban dem tay
  3. Bay cua thang do CHINH THUC hien ra tren qrels THAT

Phep 3 la ly do Phase 2 ton tai. No dung mot run doi khang — xep het cac trial
EXCLUDED len dau — roi cho thay thang do chinh thuc cham cho no diem CAO.
"""

from __future__ import annotations

import math
import sys

from src.eval import data, metrics


def _ndcg_by_hand(rels: list[int], k: int = 10) -> float:
    """gain tuyen tinh, chiet khau log2(rank+1) — quy uoc cua trec_eval."""
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def check(name: str, got: float, want: float, tol: float = 1e-9) -> bool:
    ok = abs(got - want) < tol
    print(f"  [{'DAT ' if ok else 'HONG'}] {name:44s} got={got:.10f} want={want:.10f}")
    return ok


def main() -> int:
    ok = True
    print("=" * 72)
    print("KIEM CHUNG HARNESS  (Phase 2 exit criterion)")
    print("=" * 72)

    # ---- 1 & 2: doi chieu voi ban tinh tay tren du lieu tong hop -------------
    print("\n1) nDCG@10 va contamination@10 vs ban tinh tay")
    labels = [2, 1, 0, 2, 1, 0, 0, 2, 0, 1, 2]     # thu tu da xep hang
    qrels = {"2021_1": {f"d{i}": r for i, r in enumerate(labels)}}
    run = {"2021_1": {f"d{i}": float(len(labels) - i) for i in range(len(labels))}}

    per = metrics.evaluate(run, qrels)
    ok &= check("official nDCG@10 (gain = rel)",
                per["official/ndcg_cut_10"]["2021_1"], _ndcg_by_hand(labels))
    ok &= check("eligible nDCG@10 (chi rel==2 co gain)",
                per["eligible/ndcg_cut_10"]["2021_1"],
                _ndcg_by_hand([1 if r == data.ELIGIBLE else 0 for r in labels]))
    ok &= check("contamination@10 (3 excluded trong top-10)",
                per["elig/contamination_10"]["2021_1"], 3 / 10)
    ok &= check("judged@10 (moi doc deu da cham)",
                per["elig/judged_10"]["2021_1"], 1.0)

    # ---- 3: bay cua thang do chinh thuc, tren qrels THAT --------------------
    print("\n2) Bay cua thang do chinh thuc — qrels 2021 that")
    real = data.load_qrels(2021)
    counts = data.label_counts(real)
    print(f"   {len(real)} topic | eligible={counts[2]:,} "
          f"excluded={counts[1]:,} not-relevant={counts[0]:,}")

    # Run doi khang: uu tien EXCLUDED, roi ELIGIBLE, roi phan con lai.
    # Day chinh la kieu that bai ma de tai sinh ra de loai bo.
    adversarial = {}
    for tid, docs in real.items():
        adversarial[tid] = {
            d: (3.0 if r == data.EXCLUDED else 2.0 if r == data.ELIGIBLE else 1.0)
            for d, r in docs.items()
        }
    agg = metrics.aggregate(metrics.evaluate(adversarial, real))

    print(f"\n   {'do':38s} {'gia tri':>9s}")
    for key, label in (("official/ndcg_cut_10", "nDCG@10 CHINH THUC"),
                       ("official/P_10",        "P@10 CHINH THUC"),
                       ("official/recip_rank",  "MRR CHINH THUC"),
                       ("eligible/ndcg_cut_10", "nDCG@10 eligible-only"),
                       ("eligible/P_10",        "P@10 eligible-only"),
                       ("elig/contamination_10", "Contamination@10")):
        print(f"   {label:38s} {agg[key]:9.4f}")

    hi = agg["official/ndcg_cut_10"]
    lo = agg["eligible/ndcg_cut_10"]
    con = agg["elig/contamination_10"]
    print(f"\n   He thong nay xep TOAN trial bi loai tru len dau.")
    print(f"   Thang do chinh thuc cham {hi:.3f} nDCG@10 — trong nhu mot he thong tot.")
    print(f"   Thang eligibility cham {lo:.3f}, contamination {con:.3f}.")
    ok &= check("nDCG@10 chinh thuc cao (>0.5) du ket qua vo dung",
                1.0 if hi > 0.5 else 0.0, 1.0)
    ok &= check("nDCG@10 eligible-only phai thap hon han",
                1.0 if lo < hi else 0.0, 1.0)

    print("\n" + "=" * 72)
    print("KET LUAN:", "TAT CA DAT" if ok else "CO PHEP KIEM HONG")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
