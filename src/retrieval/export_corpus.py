"""Phase 3 step 2 — export the canonical store to JSONL for Pyserini.

    python -m src.retrieval.export_corpus                 # base variant
    python -m src.retrieval.export_corpus --with-criteria # ablation variant

Pyserini expects `JsonCollection`: one document per line,
    {"id": "NCT00000102", "contents": "..."}

Two hard requirements:
1. BULK query, never a get_trial() loop. Measured on the real corpus: the
   get_trial() loop takes 2.3 minutes (5 queries/trial x 375,580); one
   GROUP_CONCAT query takes 4 seconds.
2. Contents must match store.retrieval_text() EXACTLY — that function is the
   single source of truth for "what text gets indexed". Any drift here means
   Phase 5 (dense) indexes different text than Phase 3, invalidating the
   whole ablation ladder (rung 2 would differ from rung 1 in two things, not
   one). Hence --verify, which checks a random sample.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time

from src.corpus.store import get_trial, open_db, retrieval_text

# Column order must match the join order in retrieval_text():
#   title, summary, conditions, interventions, mesh, keywords
BULK_SQL = """
SELECT t.nct_id, t.title, t.summary,
       (SELECT GROUP_CONCAT(term, '; ') FROM trial_conditions    WHERE nct_id = t.nct_id),
       (SELECT GROUP_CONCAT(term, '; ') FROM trial_interventions WHERE nct_id = t.nct_id),
       (SELECT GROUP_CONCAT(term, '; ') FROM trial_mesh          WHERE nct_id = t.nct_id),
       (SELECT GROUP_CONCAT(term, '; ') FROM trial_keywords      WHERE nct_id = t.nct_id),
       t.criteria_raw
FROM trials t
"""


def build_contents(row: sqlite3.Row | tuple, with_criteria: bool = False,
                   boost: int = 1) -> str:
    """Build the indexed text from one bulk-query row.

    boost = 1  -> original order, matches retrieval_text(get_trial(...)) exactly.
    boost > 1  -> repeats title + conditions `boost` times up front. Lucene
                  has no per-field weighting at search time, so this
                  approximates it: higher term frequency raises the BM25 score.

    with_criteria appends criteria_raw — the WINNING variant in Phase 3
    (elig nDCG@10 0.1600 -> 0.2070), contrary to store.retrieval_text()'s own note.
    """
    _, title, summary, cond, intv, mesh, kw, crit = row
    if boost > 1:
        parts = ["\n".join([title or "", cond or ""] * boost),
                 summary, intv, mesh, kw]
    else:
        parts = [title, summary, cond, intv, mesh, kw]
    if with_criteria and crit:
        parts.append(crit)
    return "\n".join(p for p in parts if p)


def verify(conn: sqlite3.Connection, n: int = 200, seed: int = 0) -> int:
    """Check the bulk query against retrieval_text() on a random sample."""
    ids = [r[0] for r in conn.execute("SELECT nct_id FROM trials")]
    random.Random(seed).shuffle(ids)
    ids = ids[:n]

    bad = 0
    for nct in ids:
        row = conn.execute(BULK_SQL + " WHERE t.nct_id = ?", (nct,)).fetchone()
        got = build_contents(row)
        want = retrieval_text(get_trial(conn, nct))
        if got != want:
            bad += 1
            if bad <= 3:
                print(f"  LECH {nct}\n    bulk: {got[:110]!r}\n    ham : {want[:110]!r}",
                      file=sys.stderr)
    print(f"Doi chieu {len(ids)} trial: lech {bad}  -> {'DAT' if bad == 0 else 'KHONG DAT'}")
    return bad


def export(conn: sqlite3.Connection, out_dir: str, with_criteria: bool,
           boost: int = 1, shard_size: int = 100_000) -> tuple[int, int]:
    """Write JSONL, sharded so Pyserini can index in parallel."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".jsonl"):
            os.remove(os.path.join(out_dir, f))

    n = empty = 0
    fh = None
    for row in conn.execute(BULK_SQL):
        if n % shard_size == 0:
            if fh:
                fh.close()
            fh = open(os.path.join(out_dir, f"docs{n // shard_size:02d}.jsonl"),
                      "w", encoding="utf-8")
        contents = build_contents(row, with_criteria, boost)
        if not contents.strip():
            # Trial has no text to index. Still written, so the index's total
            # doc count matches 375,580 — a missing doc would miscompute recall.
            empty += 1
        fh.write(json.dumps({"id": row[0], "contents": contents},
                            ensure_ascii=False) + "\n")
        n += 1
    if fh:
        fh.close()
    return n, empty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/trials.db")
    ap.add_argument("--out", default=None,
                    help="mac dinh: data/jsonl/base hoac data/jsonl/crit")
    ap.add_argument("--with-criteria", action="store_true",
                    help="them criteria_raw vao van ban index (ban de ablate)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    out = args.out or ("data/jsonl/crit" if args.with_criteria else "data/jsonl/base")
    conn = open_db(args.db)

    if not args.skip_verify and not args.with_criteria:
        if verify(conn) != 0:
            print("Dung lai: bulk query khong khop retrieval_text().", file=sys.stderr)
            return 1

    t0 = time.time()
    n, empty = export(conn, out, args.with_criteria)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"Da ghi {n:,} doc vao {out}/  ({size/1e9:.2f} GB, {time.time()-t0:.0f}s)")
    if empty:
        print(f"  luu y: {empty:,} doc khong co van ban nao (van giu de khop tong so)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
