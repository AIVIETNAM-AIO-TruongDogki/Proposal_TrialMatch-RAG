"""Phase 5 buoc 3 — benchmark 3 encoder x 2 bien the tren SUBSAMPLE.

    python -m src.dense.bench --make-subsample     # tao danh sach id mot lan
    python -m src.dense.bench --run                # ma hoa + cham diem

DIEM SUBSAMPLE BI THOI PHONG — KHONG BAO GIO DAT CANH SO CUA PHASE 3
---------------------------------------------------------------------
Subsample = 26.162 trial da cham (2021) + 20.000 nhieu ngau nhien = 46.162 doc.
Trial da cham chiem 57% subsample nhung chi 7% corpus that. It hon 8 lan nhieu
thi MOI do do deu tang. Nhung con so o day chi de XEP HANG CAC UNG VIEN VOI
NHAU, khong so voi 0.2399 cua Phase 3. So sanh voi Phase 3 chi hop le sau khi
model thang cuoc duoc ma hoa tren TOAN corpus.

De co diem tham chieu noi bo, ta build luon mot run BM25 tren DUNG 46.162 doc
do — vai giay index, va no cho biet dense hon/kem lexical TRONG CUNG dieu kien.

COT QUYET DINH LA UNION-RECALL, KHONG PHAI nDCG
------------------------------------------------
Phase 3 dat Recall@1000 = 0.4176. Do la TRAN CUNG cho moi tang phia sau:
rerank khong the cuu thu ma retrieval khong bao gio tra ve. Nen mot model
thua BM25 ve nDCG@10 nhung tim ra 15% trial lien quan ma BM25 KHONG BAO GIO
tra ve thi dang gia hon mot model hoa diem nhung tra ve cung tap tai lieu —
Phase 6 tieu thu PHAN BU, khong tieu thu diem.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

# Phai dat TRUOC moi import torch (ke ca gian tiep qua sentence-transformers).
# Lan chay dau chet OOM voi "1.73 GiB is reserved by PyTorch but unallocated" —
# phan manh, dung truong hop cai nay xu ly.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.eval import data, metrics, run_io

SUB_DIR = "data/subsample"
IDS_FILE = f"{SUB_DIR}/ids.txt"
VEC_DIR = "indexes/dense"
N_DISTRACTORS = 20_000
SEED = 0
VARIANTS = ("base", "crit")   # KHONG dung crit_x3 — xem specs/05


def make_subsample(year: int = 2021, db: str = "data/trials.db") -> int:
    """26.162 trial da cham + 20.000 nhieu ngau nhien, seed co dinh."""
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
    """Phan bu giua hai run: A tim duoc gi ma B khong, va nguoc lai.

    Chi dem trial ELIGIBLE — do la thu Phase 8 can, va la thu ca de tai do.
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
    """Run BM25 tren DUNG cung 46.162 doc — diem tham chieu noi bo."""
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


# --- Tich luy ket qua qua NHIEU tien trinh ------------------------------------
#
# Moi to hop (model, bien the) chay trong MOT TIEN TRINH RIENG: tien trinh moi =
# GPU sach tuyet doi, khong mang phan manh sang. Lan chay gop 6 to hop trong mot
# tien trinh da chet OOM ngay o to hop thu hai. Nap lai model mat ~30 giay so voi
# 16 phut ma hoa — khong dang ke.
#
# Doi lai, file ket qua phai GOP chu khong ghi de, neu khong moi tien trinh se
# xoa ket qua cua tien trinh truoc va bang tong chi con mot dong.

def results_path(year: int) -> str:
    return f"results/_dense_bench.{year}.json"


def load_rows(year: int) -> list[dict]:
    p = results_path(year)
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8"))


def merge_row(year: int, row: dict) -> list[dict]:
    """Thay dong cu cung (model, variant) roi ghi lai ca file."""
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
    # `gemini` KHONG mac dinh: quota embedding dem theo tung VAN BAN (100/phut),
    # nen ma hoa subsample mat 8,3 gio va toan corpus 67 gio — xem encode.py.
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
                # Giu lai so do cu thay vi bo trong: lan chay lai chi de cham
                # diem, nhung cot s/doc la du lieu quyet dinh cho buoc ma hoa
                # toan corpus, mat no la phai ma hoa lai chi de do gio.
                sec_per_doc = prev.get((mk, variant), {}).get("sec_per_doc")
                print(f"\n== {tag} da co vector, bo qua ma hoa ==")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            run = search.search(vec, mk, topics, 1000, args.device)
            # search() nap encoder va day ca ma tran len GPU. Khong don o day
            # thi phan manh cong don qua tung vong lap va model sau OOM — dung
            # cach lan chay dau chet o to hop thu hai.
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
