"""Phase 3 buoc 2 — xuat canonical store ra JSONL cho Pyserini.

    python -m src.retrieval.export_corpus                 # ban base
    python -m src.retrieval.export_corpus --with-criteria # ban de ablate

Pyserini nhan `JsonCollection`: moi dong mot document dang
    {"id": "NCT00000102", "contents": "..."}

HAI DIEU BAT BUOC
-----------------
1. Dung BULK query, khong lap get_trial(). Da do tren corpus that:
   vong lap get_trial() mat 2,3 phut (5 truy van/trial x 375.580);
   mot cau GROUP_CONCAT duy nhat mat 4 giay.

2. Noi dung phai KHOP CHINH XAC store.retrieval_text(). Ham do la nguon su
   that duy nhat ve "van ban nao duoc dem di index". Neu bulk query o day lech
   khoi no thi Phase 5 (dense) se index van ban KHAC Phase 3, va ca thang
   ablation mat hieu luc — bac 2 se khac bac 1 o hai thu cung luc thay vi mot.
   Vi vay co --verify doi chieu tren mau ngau nhien.
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

# Thu tu cot phai trung thu tu ghep trong retrieval_text():
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
    """Ghep van ban index tu mot dong bulk query.

    boost = 1  -> thu tu goc, khop y het retrieval_text(get_trial(...)).
    boost > 1  -> lap lai title + conditions `boost` lan va dua len dau.
                  Lucene khong cho dat trong so truong luc search, nen day la
                  cach xap xi chuan: tang tan suat term = tang diem BM25.

    with_criteria them criteria_raw vao cuoi. Do la bien the THANG o Phase 3
    (elig nDCG@10 0.1600 -> 0.2070), trai voi ghi chu trong store.retrieval_text().
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
    """Doi chieu bulk query voi retrieval_text() tren mau ngau nhien."""
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
    """Ghi JSONL, chia nhieu shard de Pyserini index song song."""
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
            # Trial khong co van ban nao de index. Van ghi ra de tong so doc
            # trong index khop 375.580 — thieu doc thi recall bi tinh sai.
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
                    help="them criteria_raw vao van ban index")
    ap.add_argument("--boost", type=int, default=1,
                    help="lap lai title+conditions N lan (mac dinh 1 = khong boost)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    # Ten thu muc suy tu cau hinh, de khong bao gio lan bien the voi nhau.
    variant = ("crit" if args.with_criteria else "base") + \
              (f"_x{args.boost}" if args.boost > 1 else "")
    out = args.out or f"data/jsonl/{variant}"
    conn = open_db(args.db)

    if not args.skip_verify and not args.with_criteria and args.boost == 1:
        if verify(conn) != 0:
            print("Dung lai: bulk query khong khop retrieval_text().", file=sys.stderr)
            return 1

    t0 = time.time()
    n, empty = export(conn, out, args.with_criteria, args.boost)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"Da ghi {n:,} doc vao {out}/  ({size/1e9:.2f} GB, {time.time()-t0:.0f}s)")
    if empty:
        print(f"  luu y: {empty:,} doc khong co van ban nao (van giu de khop tong so)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
