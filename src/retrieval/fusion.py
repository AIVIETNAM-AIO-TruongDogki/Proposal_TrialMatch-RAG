"""Phase 6 — hop nhat hai chan truy hoi (bac 3 cua thang ablation).

    python -m src.retrieval.fusion --lexical runs/bm25_best.dev.txt \
        --dense runs/dense.dev.txt --out runs/hybrid.dev.txt
    python -m src.retrieval.fusion ... --method wsum --tune

HAI PHUONG PHAP, BAO CAO CA HAI
--------------------------------
RRF:   score(d) = sum_i 1/(k + rank_i(d)),  k = 60.
       Mot dong, khong can chuan hoa diem, va thuong THANG weighted fusion.
       Vi vay no la mac dinh, khong phai phuong an du phong.

wsum:  chuan hoa diem (minmax hoac zscore) roi cong co trong so.
       Nhay cam voi thang diem: BM25 tra ve diem Lucene khong chan tren, dense
       tra ve cosine trong [-1,1]. Khong chuan hoa thi mot chan nuot chan kia.

PHAN BU MOI LA KET QUA, KHONG PHAI DIEM HOP NHAT
-------------------------------------------------
Cau hoi cua Phase 6 khong phai "diem hop nhat cao hon bao nhieu" ma "hai chan
co that su mang tin hieu KHAC NHAU khong". Neu chong lan gan hoan toan thi
viec hop nhat mua duoc rat it, va luan diem "ca hai chan deu chiu luc" cua de
xuat phai duoc noi lai cho dung — do la mot ket qua phai BAO CAO, khong phai
mot that bai phai giau.
"""

from __future__ import annotations

import argparse
import sys

from src.eval import data, metrics, run_io

RRF_K = 60


def _ranked(run_topic: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(run_topic.items(), key=lambda kv: (-kv[1], kv[0]))]


def rrf(runs: list[dict], k: int = RRF_K, weights: list[float] | None = None
        ) -> dict[str, dict[str, float]]:
    weights = weights or [1.0] * len(runs)
    out: dict[str, dict[str, float]] = {}
    for w, run in zip(weights, runs):
        for tid, docs in run.items():
            acc = out.setdefault(tid, {})
            for rank, doc in enumerate(_ranked(docs), 1):
                acc[doc] = acc.get(doc, 0.0) + w / (k + rank)
    return out


def _norm(scores: dict[str, float], how: str) -> dict[str, float]:
    if not scores:
        return {}
    v = list(scores.values())
    if how == "minmax":
        lo, hi = min(v), max(v)
        rng = hi - lo
        return {d: (s - lo) / rng if rng else 0.0 for d, s in scores.items()}
    mu = sum(v) / len(v)
    sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
    return {d: (s - mu) / sd if sd else 0.0 for d, s in scores.items()}


def wsum(runs: list[dict], weights: list[float], how: str = "minmax"
         ) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for w, run in zip(weights, runs):
        for tid, docs in run.items():
            acc = out.setdefault(tid, {})
            for doc, s in _norm(docs, how).items():
                acc[doc] = acc.get(doc, 0.0) + w * s
    return out


def complementarity(lex: dict, dense: dict, qrels: dict, k: int = 1000) -> dict:
    """Bang phan bu — deliverable BAT BUOC cua Phase 6, khong phai tuy chon.

    Chi dem trial ELIGIBLE: do la thu Phase 8 can va thu de tai do.
    """
    only_l = only_d = both = neither = gold = 0
    for tid, docs in qrels.items():
        g = {d for d, r in docs.items() if r == data.ELIGIBLE}
        if not g:
            continue
        L = set(_ranked(lex.get(tid, {}))[:k])
        D = set(_ranked(dense.get(tid, {}))[:k])
        gold += len(g)
        both += len(g & L & D)
        only_l += len(g & L - D)
        only_d += len(g & D - L)
        neither += len(g - L - D)
    n = max(gold, 1)
    return {"gold": gold, "both": both / n, "only_lexical": only_l / n,
            "only_dense": only_d / n, "neither": neither / n,
            "union": (both + only_l + only_d) / n}


def print_complementarity(c: dict, k: int) -> None:
    print(f"\nPHAN BU @{k} (chi trial ELIGIBLE, {c['gold']:,} trial vang)")
    print("-" * 60)
    print(f"  ca hai chan tim duoc      {c['both']:.4f}")
    print(f"  CHI lexical tim duoc      {c['only_lexical']:.4f}")
    print(f"  CHI dense tim duoc        {c['only_dense']:.4f}")
    print(f"  khong chan nao tim duoc   {c['neither']:.4f}   <- tran cung cua ca he")
    print(f"  hop (union recall)        {c['union']:.4f}")
    share = c["only_dense"] / c["union"] if c["union"] else 0.0
    print(f"\n  dense dong gop rieng {share*100:.1f}% cua union.")
    if share < 0.05:
        print("  -> Chong lan gan hoan toan: hop nhat mua duoc RAT IT. Luan diem")
        print("     'ca hai chan deu chiu luc' can duoc noi lai cho dung.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="rrf", choices=["rrf", "wsum"])
    ap.add_argument("--norm", default="minmax", choices=["minmax", "zscore"])
    ap.add_argument("--weight", type=float, default=0.5,
                    help="trong so cua chan LEXICAL (dense = 1-w)")
    ap.add_argument("--rrf-k", type=int, default=RRF_K)
    ap.add_argument("--tune", action="store_true",
                    help="quet trong so tren dev, in bang")
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! TAP TEST 2022 — chi cham MOT LAN o Phase 11.", file=sys.stderr)

    lex = run_io.read_run(args.lexical)
    den = run_io.read_run(args.dense)
    qrels = data.load_qrels(args.year)
    print(f"lexical: {len(lex)} topic   dense: {len(den)} topic")

    print_complementarity(complementarity(lex, den, qrels, args.depth), args.depth)

    if args.tune:
        print(f"\nQUET TRONG SO ({args.method}, chuan hoa={args.norm})")
        print(f"{'w_lex':>6s} {'eligNDCG10':>11s} {'officialNDCG10':>15s} {'rec@1000':>9s}")
        best = (None, -1.0)
        for w in [i / 10 for i in range(11)]:
            r = (rrf([lex, den], args.rrf_k, [w, 1 - w]) if args.method == "rrf"
                 else wsum([lex, den], [w, 1 - w], args.norm))
            a = metrics.aggregate(metrics.evaluate(r, qrels))
            print(f"{w:6.1f} {a['eligible/ndcg_cut_10']:11.4f} "
                  f"{a['official/ndcg_cut_10']:15.4f} {a['elig/recall_1000']:9.4f}")
            if a["eligible/ndcg_cut_10"] > best[1]:
                best = (w, a["eligible/ndcg_cut_10"])
        print(f"\nTot nhat: w_lex={best[0]} (elig nDCG@10={best[1]:.4f})")
        args.weight = best[0]

    w = args.weight
    run = (rrf([lex, den], args.rrf_k, [w, 1 - w]) if args.method == "rrf"
           else wsum([lex, den], [w, 1 - w], args.norm))
    tag = args.tag or args.out.split("/")[-1].rsplit(".", 1)[0]
    run_io.write_run(args.out, run, tag, depth=args.depth)
    print(f"\nDa ghi {args.out}  (method={args.method}, w_lex={w})")
    print(f"\nCham diem:\n  PYTHONPATH=. .venv/bin/python -m src.eval.score "
          f"{args.out} --year {args.year} --vs results/bm25_best.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
