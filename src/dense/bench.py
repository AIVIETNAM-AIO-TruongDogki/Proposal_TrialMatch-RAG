"""Phase 5 step 3 — benchmark 3 encoders x 2 variants on a SUBSAMPLE.

    python -m src.dense.bench --make-subsample     # build the id list once
    python -m src.dense.bench --run                # encode + score

Subsample scores are INFLATED — never compare them against Phase 3's number.
Subsample = 26,162 judged trials (2021) + 20,000 random distractors = 46,162
docs. Judged trials are 57% of the subsample but only 7% of the real corpus —
8x less noise inflates every metric. These numbers only RANK candidates
against each other, not against Phase 3's 0.2399; that comparison is only
valid once the winning model is encoded on the FULL corpus. For an internal
reference point, a BM25 run is also built on the SAME 46,162 docs — seconds
to index, and shows dense vs. lexical under identical conditions.

The deciding column is UNION-RECALL, not nDCG. Phase 3 measured
Recall@1000 = 0.4176 — a hard ceiling for every later stage: reranking can't
save a trial retrieval never returns. So a model that loses on nDCG@10 but
finds 15% of relevant trials BM25 never returns is worth more than a model
that scores higher while returning the same document set — Phase 6 consumes
COMPLEMENTARITY, not score.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

# Must be set BEFORE any torch import (including indirectly via
# sentence-transformers). The first run OOM'd with "1.73 GiB is reserved by
# PyTorch but unallocated" — fragmentation, exactly what this setting fixes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.eval import data, metrics, run_io

SUB_DIR = "data/subsample"
IDS_FILE = f"{SUB_DIR}/ids.txt"
VEC_DIR = "indexes/dense"
N_DISTRACTORS = 20_000
SEED = 0
VARIANTS = ("base", "crit")   # KHONG dung crit_x3 — xem specs/05


def make_subsample(year: int = 2021, db: str = "data/trials.db") -> int:
    """26,162 judged trials + 20,000 random distractors, fixed seed."""
    import sqlite3
    qrels = data.load_qrels(year)
    judged = {d for v in qrels.values() for d in v}
    conn = sqlite3.connect(db)
    all_ids = [r[0] for r in conn.execute("SELECT nct_id FROM trials")]
    pool = [i for i in all_ids if i not in judged]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    keep = sorted(judged | set(pool[:N_DISTRACTORS]))

    os.makedirs(SUB_DIR, exist_ok=True)
    with open(IDS_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(keep) + "\n")
    print(f"{len(judged):,} da cham + {N_DISTRACTORS:,} nhieu = {len(keep):,} doc "
          f"({len(keep)/len(all_ids)*100:.1f}% corpus)")
    print(f"Da ghi {IDS_FILE}")
    return 0


def union_recall(run_a: dict, run_b: dict, qrels: dict, k: int = 1000) -> dict:
    """Complementarity between two runs: what A finds that B doesn't, and vice versa.

    Counts only ELIGIBLE trials — what Phase 8 needs, and what this project measures.
    """
    only_a = only_b = both = gold_tot = 0
    for tid, docs in qrels.items():
        gold = {d for d, r in docs.items() if r == data.ELIGIBLE}
        if not gold:
            continue
        A = set(sorted(run_a.get(tid, {}), key=lambda d: -run_a[tid][d])[:k])
        B = set(sorted(run_b.get(tid, {}), key=lambda d: -run_b[tid][d])[:k])
        gold_tot += len(gold)
        both += len(gold & A & B)
        only_a += len(gold & A - B)
        only_b += len(gold & B - A)
    n = max(gold_tot, 1)
    return {"gold": gold_tot, "both": both / n, "only_a": only_a / n,
            "only_b": only_b / n, "union": (both + only_a + only_b) / n}


def bm25_subsample_run(year: int, out: str) -> str:
    """Run BM25 on the SAME 46,162 docs — internal reference point."""
    if os.path.exists(out):
        return out
    idx = "indexes/bm25-sub"
    jsonl = f"{SUB_DIR}/jsonl"
    if not os.path.isdir(idx):
        if not os.path.isdir(jsonl):
            keep = {l.strip() for l in open(IDS_FILE, encoding="utf-8") if l.strip()}
            os.makedirs(jsonl, exist_ok=True)
            n = 0
            with open(f"{jsonl}/docs00.jsonl", "w", encoding="utf-8") as w:
                for f in sorted(os.listdir("data/jsonl/crit_fields")):
                    if not f.endswith(".jsonl"):
                        continue
                    for line in open(f"data/jsonl/crit_fields/{f}", encoding="utf-8"):
                        o = json.loads(line)
                        if o["id"] in keep:
                            w.write(line)
                            n += 1
            print(f"  subsample jsonl: {n:,} doc")
        subprocess.run([sys.executable, "-m", "pyserini.index.lucene",
                        "--collection", "JsonCollection", "--input", jsonl,
                        "--index", idx, "--generator",
                        "DefaultLuceneDocumentGenerator", "--threads", "16"],
                       check=True, capture_output=True)
    from src.retrieval import bm25
    topics = data.load_topics(year)
    run = bm25.search(idx, topics, 1.8, 1.0, 1000)
    run_io.write_run(out, run, "bm25_sub")
    return out


# --- Accumulate results ACROSS PROCESSES --------------------------------------
#
# Each (model, variant) pair runs in ITS OWN PROCESS: a fresh process means a
# truly clean GPU, no carried-over fragmentation. Bundling 6 pairs into one
# process OOM'd on the second pair. Reloading a model costs ~30s against
# 16min of encoding — negligible.
#
# In exchange, the results file must be MERGED, not overwritten, or each
# process would erase the previous one's results, leaving only one row.

def results_path(year: int) -> str:
    return f"results/_dense_bench.{year}.json"


def load_rows(year: int) -> list[dict]:
    p = results_path(year)
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8"))


def merge_row(year: int, row: dict) -> list[dict]:
    """Replace the existing (model, variant) row and rewrite the whole file."""
    rows = [r for r in load_rows(year)
            if (r["model"], r["variant"]) != (row["model"], row["variant"])]
    rows.append(row)
    p = results_path(year)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rows, open(p, "w", encoding="utf-8"), indent=2)
    return rows


def report(year: int, bm_agg: dict | None = None) -> None:
    rows = load_rows(year)
    if not rows:
        print(f"Chua co ket qua nao trong {results_path(year)}.", file=sys.stderr)
        return
    print("\n" + "=" * 96)
    print("BENCHMARK SUBSAMPLE — diem BI THOI PHONG, chi de xep hang voi nhau")
    print("=" * 96)
    hdr = (f"{'model':10s} {'bien the':11s} {'eligNDCG10':>11s} {'rec@1000':>9s} "
           f"{'contam10':>9s} {'chi dense':>10s} {'chi bm25':>9s} {'union':>7s} {'s/doc':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["union"]):
        spd = r.get("sec_per_doc")
        spd_s = "     —" if spd is None else f"{spd:7.4f}"
        print(f"{r['model']:10s} {r['variant']:11s} {r['elig_ndcg10']:11.4f} "
              f"{r['recall1000']:9.4f} {r['contam10']:9.4f} {r['only_dense']:10.4f} "
              f"{r['only_bm25']:9.4f} {r['union']:7.4f} {spd_s}")
    if bm_agg:
        print(f"\n{'BM25 (cung subsample)':30s} eligNDCG10={bm_agg['eligible/ndcg_cut_10']:.4f} "
              f"recall@1000={bm_agg['elig/recall_1000']:.4f}")
    print("\n'chi dense' = ty le trial ELIGIBLE ma dense tim ra con BM25 KHONG.")
    print("Do la cot quyet dinh: Phase 6 tieu thu phan bu, khong tieu thu diem.")
    print("\nDu phong ma hoa toan corpus (375.580 doc):")
    for r in sorted(rows, key=lambda x: (x["model"], x["variant"])):
        spd = r.get("sec_per_doc")
        if spd:
            print(f"  {r['model']:10s} {r['variant']:11s} {spd*375580/60:6.1f} phut")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-subsample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="chi in bang tong tu ket qua da tich luy, khong chay gi")
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021])
    # `gemini` not in the default: embedding quota counts per TEXT (100/min),
    # so encoding the subsample takes 8.3h and the full corpus 67h — see encode.py.
    ap.add_argument("--models", default="bge-m3,qwen3,medcpt")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.make_subsample:
        return make_subsample(args.year)
    if args.report:
        bm = "runs/_sub_bm25.dev.txt"
        agg = (metrics.aggregate(metrics.evaluate(run_io.read_run(bm),
                                                  data.load_qrels(args.year)))
               if os.path.exists(bm) else None)
        report(args.year, agg)
        return 0
    if not args.run:
        ap.error("can --make-subsample, --run hoac --report")
    if not os.path.exists(IDS_FILE):
        print(f"Chua co {IDS_FILE}. Chay --make-subsample truoc.", file=sys.stderr)
        return 1

    from src.dense import encode, search

    qrels = data.load_qrels(args.year)
    topics = data.load_topics(args.year)
    print("Run BM25 tham chieu tren cung subsample...")
    bm_path = bm25_subsample_run(args.year, f"runs/_sub_bm25.dev.txt")
    bm_run = run_io.read_run(bm_path)

    os.makedirs(VEC_DIR, exist_ok=True)
    import torch
    prev = {(r["model"], r["variant"]): r for r in load_rows(args.year)}
    for variant in args.variants.split(","):
        for mk in args.models.split(","):
            tag = f"{mk}.{variant}.sub"
            vec = f"{VEC_DIR}/{tag}.npz"
            if not os.path.exists(vec):
                print(f"\n== ma hoa {tag} ==")
                keep = {l.strip() for l in open(IDS_FILE, encoding="utf-8") if l.strip()}
                ids, texts = encode.read_jsonl_dir(f"data/jsonl/{variant}", keep)
                enc = encode.load_encoder(mk, args.device)
                t0 = time.time()
                V, owner = encode.encode_docs(enc, ids, texts)
                sec_per_doc = (time.time() - t0) / max(len(ids), 1)
                encode.save(vec, ids, V, owner, encode.MODELS[mk])
                del enc, V, owner
            else:
                # Keep the old timing instead of leaving it blank: a rerun
                # that's only rescoring shouldn't lose the s/doc data the
                # full-corpus encoding decision depends on, or it would have
                # to re-encode just to time it.
                sec_per_doc = prev.get((mk, variant), {}).get("sec_per_doc")
                print(f"\n== {tag} da co vector, bo qua ma hoa ==")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            run = search.search(vec, mk, topics, 1000, args.device)
            # search() loads the encoder and pushes the whole matrix to the
            # GPU. Without clearing here, fragmentation accumulates across
            # loop iterations and the next model OOMs — exactly how the first
            # run died on the second pair.
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            run_path = f"runs/_sub_{tag}.dev.txt"
            run_io.write_run(run_path, run, tag)

            per = metrics.evaluate(run, qrels)
            agg = metrics.aggregate(per)
            ur = union_recall(run, bm_run, qrels)
            merge_row(args.year, {
                "model": mk, "variant": variant,
                "elig_ndcg10": agg["eligible/ndcg_cut_10"],
                "recall1000": agg["elig/recall_1000"],
                "contam10": agg["elig/contamination_10"],
                "only_dense": ur["only_a"], "only_bm25": ur["only_b"],
                "union": ur["union"], "sec_per_doc": sec_per_doc})
            print(f"  elig nDCG@10={agg['eligible/ndcg_cut_10']:.4f}  "
                  f"recall@1000={agg['elig/recall_1000']:.4f}  "
                  f"chi dense={ur['only_a']:.4f}  union={ur['union']:.4f}")

    report(args.year, metrics.aggregate(metrics.evaluate(bm_run, qrels)))
    print(f"\nDa ghi {results_path(args.year)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
