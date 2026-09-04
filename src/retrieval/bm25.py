"""Phase 3 step 4 — BM25 retrieval, writes a TREC-format run file.

    python -m src.retrieval.bm25 --index indexes/bm25-base --out runs/bm25.dev.txt
    python -m src.retrieval.bm25 --k1 1.2 --b 0.75 --query-mode nodeid ...

Rung 1's principle: the query is the RAW patient narrative, verbatim. No
entity extraction, no query expansion, no term normalization — that's Phase
4; mixing it in here would change two things at once between rung 1 and
rung 2, and an ablation couldn't attribute the gain to either. The only
exception is Lucene special-character escaping, which is mechanically
required, and every query transformation is counted and printed.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from src.eval import data, run_io

# Characters Lucene treats as query syntax. Unescaped, a narrative containing
# "[**2148-10-1**]" breaks the query parser.
LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

# De-identification marker, e.g. "[**2148-10-1**]" — an artifact, not clinical content.
DEID = re.compile(r"\[\*\*.*?\*\*\]")


def prepare_query(text: str, mode: str = "raw") -> str:
    """mode='raw'    : verbatim, only Lucene special chars escaped
       mode='nodeid' : strips de-id markers before escaping
    """
    if mode == "nodeid":
        text = DEID.sub(" ", text)
    text = LUCENE_SPECIAL.sub(r" ", text)
    return " ".join(text.split())


def search(index_dir: str, topics: dict[str, str], k1: float, b: float,
           depth: int = 1000, mode: str = "raw", threads: int = 16
           ) -> dict[str, dict[str, float]]:
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1, b)

    qids = sorted(topics)
    queries = [prepare_query(topics[q], mode) for q in qids]

    n_changed = sum(1 for q, raw in zip(queries, (topics[i] for i in qids))
                    if q != " ".join(raw.split()))
    print(f"  {len(qids)} truy van, {n_changed} bi doi khi escape/xoa deid "
          f"(mode={mode})")

    hits = searcher.batch_search(queries, qids, k=depth, threads=threads)
    return {qid: {h.docid: float(h.score) for h in hs} for qid, hs in hits.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="indexes/bm25-base")
    ap.add_argument("--out", default="runs/bm25.dev.txt")
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--k1", type=float, default=0.9, help="mac dinh Lucene")
    ap.add_argument("--b", type=float, default=0.4, help="mac dinh Lucene")
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--query-mode", default="raw", choices=["raw", "nodeid"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! Dang chay tren TAP TEST 2022 — chi duoc cham MOT LAN o Phase 11.",
              file=sys.stderr)

    topics = data.load_topics(args.year)
    print(f"Nap {len(topics)} topic nam {args.year}")
    print(f"Index: {args.index}   k1={args.k1} b={args.b}")

    t0 = time.time()
    run = search(args.index, topics, args.k1, args.b, args.depth,
                 args.query_mode, args.threads)
    el = time.time() - t0

    empty = [q for q in topics if not run.get(q)]
    if empty:
        print(f"CANH BAO: {len(empty)} topic khong tra ve ket qua nao: {empty[:5]}",
              file=sys.stderr)

    tag = args.tag or args.out.split("/")[-1].rsplit(".", 1)[0]
    run_io.write_run(args.out, run, tag, depth=args.depth)
    avg = sum(len(v) for v in run.values()) / max(len(run), 1)
    print(f"Da ghi {args.out}  ({el:.1f}s, trung binh {avg:.0f} ket qua/topic)")
    print(f"\nCham diem:\n  PYTHONPATH=. .venv/bin/python -m src.eval.score "
          f"{args.out} --year {args.year} --vs results/_random.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
