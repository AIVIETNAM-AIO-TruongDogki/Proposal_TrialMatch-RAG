"""Phase 1 — build the canonical store: 375,580 XML files -> SQLite.

    python -m src.corpus.build_db --db data/trials.db

Workers parse in parallel; the main process is the sole SQLite writer. Ends
with a parse-quality report — that report IS Phase 1's deliverable, not an
appendix: without it there's no way to know if 5 million criteria rows are correct.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter
from multiprocessing import Pool

from src.corpus.parse import Trial, parse_trial

SCHEMA = os.path.join(os.path.dirname(__file__), "schema.sql")
BATCH = 2000


def iter_paths(rawdata: str):
    """Walk the corpus tree. Only trial files (NCT*.xml) — topics*.xml also
    lives in rawdata/, so a bare `*.xml` glob would overcount by 2 files."""
    for root, _, files in os.walk(rawdata):
        for f in files:
            if f.startswith("NCT") and f.endswith(".xml"):
                yield os.path.join(root, f)


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    with open(SCHEMA, encoding="utf-8") as fh:
        conn.executescript(fh.read())
    return conn


def write_batch(conn: sqlite3.Connection, batch: list[Trial]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO trials (nct_id,title,summary,detail,gender,"
        "min_age_years,max_age_years,min_age_raw,max_age_raw,healthy_volunteers,"
        "phase,status,study_type,criteria_raw,parse_method,n_criteria) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(t.nct_id, t.title, t.summary, t.detail, t.gender, t.min_age_years,
          t.max_age_years, t.min_age_raw, t.max_age_raw, t.healthy_volunteers,
          t.phase, t.status, t.study_type, t.criteria_raw, t.parse_method,
          len(t.criteria)) for t in batch],
    )
    conn.executemany(
        "INSERT INTO criteria (nct_id,idx,section,text,span_start,span_end,lead_in) "
        "VALUES (?,?,?,?,?,?,?)",
        [(t.nct_id, c.idx, c.section, c.text, c.span_start, c.span_end, c.lead_in)
         for t in batch for c in t.criteria],
    )
    for table, rows in (
        ("trial_conditions",    [(t.nct_id, x) for t in batch for x in t.conditions]),
        ("trial_keywords",      [(t.nct_id, x) for t in batch for x in t.keywords]),
    ):
        conn.executemany(f"INSERT INTO {table} (nct_id,term) VALUES (?,?)", rows)
    conn.executemany(
        "INSERT INTO trial_interventions (nct_id,term,itype) VALUES (?,?,?)",
        [(t.nct_id, n, ty) for t in batch for n, ty in t.interventions],
    )
    conn.executemany(
        "INSERT INTO trial_mesh (nct_id,term,source) VALUES (?,?,?)",
        [(t.nct_id, term, src) for t in batch for term, src in t.mesh],
    )


def quality_report(conn: sqlite3.Connection) -> None:
    """Parse-quality report — Phase 1's exit criterion."""
    q = lambda sql: conn.execute(sql).fetchone()[0]

    total = q("SELECT COUNT(*) FROM trials")
    with_blob = q("SELECT COUNT(*) FROM trials WHERE criteria_raw IS NOT NULL")
    n_crit = q("SELECT COUNT(*) FROM criteria")
    with_any = q("SELECT COUNT(*) FROM trials WHERE n_criteria > 0")
    with_inc = q("SELECT COUNT(DISTINCT nct_id) FROM criteria WHERE section='inclusion'")
    with_exc = q("SELECT COUNT(DISTINCT nct_id) FROM criteria WHERE section='exclusion'")

    pct = lambda a, b: f"{a/b*100:5.1f}%" if b else "  n/a"
    print("\n" + "=" * 62)
    print("BAO CAO CHAT LUONG PARSE  (Phase 1 deliverable)")
    print("=" * 62)
    print(f"  Trial                       : {total:>9,}")
    print(f"  Co eligibility blob         : {with_blob:>9,}  {pct(with_blob, total)}")
    print(f"  Tach duoc >=1 criterion     : {with_any:>9,}  {pct(with_any, with_blob)} cua so co blob")
    print(f"  Co >=1 inclusion            : {with_inc:>9,}  {pct(with_inc, with_blob)}")
    print(f"  Co >=1 exclusion            : {with_exc:>9,}  {pct(with_exc, with_blob)}")
    print(f"  Tong criteria               : {n_crit:>9,}   ({n_crit/max(with_any,1):.1f}/trial)")

    print("\n  Phuong phap tach:")
    for m, c in conn.execute(
        "SELECT parse_method, COUNT(*) FROM trials GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"    {m or '(null)':16s} {c:>9,}  {pct(c, total)}")

    print("\n  Criteria/trial (trial co criteria):")
    rows = [r[0] for r in conn.execute(
        "SELECT n_criteria FROM criteria_counts_v")]
    if rows:
        print(f"    min={rows[0]}  p50={rows[len(rows)//2]}  "
              f"p95={rows[int(len(rows)*0.95)]}  max={rows[-1]}")

    print("\n  Bo loc co cau truc (NULL = trial khong khai bao):")
    for col in ("gender", "min_age_years", "max_age_years", "healthy_volunteers"):
        nn = q(f"SELECT COUNT(*) FROM trials WHERE {col} IS NOT NULL")
        print(f"    {col:20s} co gia tri: {nn:>9,}  {pct(nn, total)}")
    na = q("SELECT COUNT(*) FROM trials WHERE min_age_years IS NULL AND min_age_raw IS NOT NULL")
    print(f"    min_age khong parse duoc (vd 'N/A'): {na:,}")

    # Span verification: the technical basis of invariant 3.
    bad = 0
    checked = 0
    for raw, text, s, e in conn.execute(
        "SELECT t.criteria_raw, c.text, c.span_start, c.span_end "
        "FROM criteria c JOIN trials t USING(nct_id) "
        "WHERE c.nct_id IN (SELECT nct_id FROM trials ORDER BY RANDOM() LIMIT 3000)"
    ):
        checked += 1
        if " ".join(raw[s:e].split()) != text:
            bad += 1
    print(f"\n  Kiem chung span (mau {checked:,}): sai {bad}  "
          f"-> {'DAT' if bad == 0 else 'KHONG DAT'}")
    print("=" * 62)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdata", default="rawdata")
    ap.add_argument("--db", default="data/trials.db")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="chi parse N file (de test)")
    args = ap.parse_args()

    print(f"Duyet {args.rawdata} ...", flush=True)
    paths = list(iter_paths(args.rawdata))
    if args.limit:
        paths = paths[: args.limit]
    print(f"  {len(paths):,} file trial")

    if os.path.exists(args.db):
        os.remove(args.db)
    conn = connect(args.db)

    t0 = time.time()
    batch: list[Trial] = []
    done = failed = 0
    with Pool(args.workers) as pool:
        for t in pool.imap_unordered(parse_trial, paths, chunksize=200):
            done += 1
            if t is None:
                failed += 1
            else:
                batch.append(t)
            if len(batch) >= BATCH:
                write_batch(conn, batch)
                conn.commit()
                batch.clear()
            if done % 25000 == 0:
                el = time.time() - t0
                print(f"  {done:>7,}/{len(paths):,}  {el:5.0f}s  "
                      f"({done/el:,.0f}/s)", flush=True)
    if batch:
        write_batch(conn, batch)
        conn.commit()

    conn.execute("CREATE TEMP VIEW criteria_counts_v AS "
                 "SELECT n_criteria FROM trials WHERE n_criteria > 0 ORDER BY n_criteria")
    print(f"\nXong {done:,} file trong {time.time()-t0:.0f}s  (hong: {failed})")
    quality_report(conn)
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    print(f"\nDB: {args.db}  ({os.path.getsize(args.db)/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
