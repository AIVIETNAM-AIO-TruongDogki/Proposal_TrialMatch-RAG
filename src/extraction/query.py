"""Phase 4 buoc 4 — ho so -> truy van BM25, va do lai voi Phase 3.

    python -m src.extraction.query --model qwen2.5:7b-instruct --mode prof

BA BIEN THE
-----------
  prof        chi cac term da trich (conditions + biomarkers + treatments + comorbidities)
  prof_narr   benh an goc + term da trich (noi them)
  hyde        sinh mot mo ta thu nghiem gia dinh tu ho so, truy hoi bang no

DIEU KIEN DE ABLATION HOP LE
----------------------------
Phai chay tren DUNG cau hinh thang cuoc cua Phase 3: index `bm25-critfields`,
k1=1.8, b=1.0. Doi index hay doi tham so la doi hai thu cung luc, va chenh lech
do se khong quy duoc cho ai.

BAY PHU DINH — DOI XUNG VOI BAY DA GAP O PHASE 3
------------------------------------------------
Term bi phu dinh KHONG duoc vao truy van. "no history of diabetes" ma nem
"diabetes" vao truy van thi BM25 se keo ve dung nhung trial noi ve benh ma
benh nhan KHONG co. Do la phien ban nguoc cua dieu Phase 3 da phat hien: BM25
khong doc duoc phu dinh, ca o phia tai lieu lan o phia truy van.

Nhung chung VAN NAM trong ho so — Phase 8 can chung de ket luan `satisfied`
cho mot tieu chi loai tru. Loai khoi TRUY VAN, giu trong HO SO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.eval import data, run_io
from src.extraction import ollama, schema

BEST_INDEX = "indexes/bm25-critfields"   # cau hinh thang cuoc Phase 3
BEST_K1, BEST_B = 1.8, 1.0

HYDE_SYSTEM = """\
You write a short, plausible clinical trial description for which the given \
patient would be a strong candidate. Write 3-5 sentences in the style of a \
ClinicalTrials.gov brief summary: condition studied, target population, and \
intervention type. Use standard medical terminology. Do not mention the \
patient, and do not invent specific trial names or NCT numbers.
"""


def profile_terms(profile: dict, include_negated: bool = False) -> list[str]:
    """Term dua vao truy van. Mac dinh BO term bi phu dinh — xem docstring."""
    out: list[str] = []
    for f in schema.QUERY_FIELDS:
        for it in profile.get(f) or []:
            if it.get("status") == "negated" and not include_negated:
                continue
            name = (it.get("name") or "").strip()
            if name:
                out.append(name)
    # Bo trung lap, giu thu tu — thu tu khong doi diem BM25 nhung giup doc log.
    seen: set[str] = set()
    return [t for t in out if not (t.lower() in seen or seen.add(t.lower()))]


def build_query(profile: dict, narrative: str, mode: str,
                hyde_text: str | None = None) -> str:
    terms = profile_terms(profile)
    if mode == "prof":
        return "; ".join(terms)
    if mode == "prof_narr":
        return narrative + "\n" + "; ".join(terms)
    if mode == "hyde":
        return hyde_text or narrative
    raise ValueError(mode)


def load_profiles(model: str, year: int, profile_dir: str = "data/profiles") -> dict:
    path = os.path.join(profile_dir, f"{year}.{model.replace(':', '_')}.json")
    if not os.path.exists(path):
        raise SystemExit(f"Chua co {path}. Chay src.extraction.extract truoc.")
    return json.load(open(path, encoding="utf-8"))["records"]


def gen_hyde(model: str, profile: dict, topics_id: str, cache: dict) -> str:
    if topics_id in cache:
        return cache[topics_id]
    terms = profile_terms(profile)
    desc = "; ".join(terms) or "(no extracted findings)"
    age = (profile.get("age") or {}).get("value")
    sex = (profile.get("sex") or {}).get("value")
    who = f"{age}-year-old {sex}" if age and sex else "patient"
    prof_txt = f"A {who} with: {desc}"
    body = ollama.chat_json(
        model, HYDE_SYSTEM,
        prof_txt + "\n\nWrite the trial description as JSON {\"text\": \"...\"}.",
        {"type": "object", "properties": {"text": {"type": "string"}},
         "required": ["text"]})[0]
    cache[topics_id] = (body or {}).get("text", "") or prof_txt
    return cache[topics_id]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model da dung o extract")
    ap.add_argument("--mode", default="prof", choices=["prof", "prof_narr", "hyde"])
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--index", default=BEST_INDEX)
    ap.add_argument("--k1", type=float, default=BEST_K1)
    ap.add_argument("--b", type=float, default=BEST_B)
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if (args.index, args.k1, args.b) != (BEST_INDEX, BEST_K1, BEST_B):
        print("!! Khong dung cau hinh thang cuoc cua Phase 3 — so sanh voi "
              "bm25_best se KHONG hop le (doi hai thu cung luc).", file=sys.stderr)

    from src.retrieval import bm25

    topics = data.load_topics(args.year)
    recs = load_profiles(args.model, args.year)

    hyde_cache: dict[str, str] = {}
    queries: dict[str, str] = {}
    n_empty = 0
    for tid, narrative in topics.items():
        prof = (recs.get(tid) or {}).get("clean")
        if not prof:
            # Trich xuat hong: lui ve benh an goc thay vi bo topic. Bo topic se
            # lam diem trung binh dep len mot cach gia tao.
            queries[tid] = narrative
            n_empty += 1
            continue
        hyde = gen_hyde(args.model, prof, tid, hyde_cache) if args.mode == "hyde" else None
        q = build_query(prof, narrative, args.mode, hyde)
        if not q.strip():
            q, n_empty = narrative, n_empty + 1
        queries[tid] = q

    print(f"{len(queries)} truy van che do '{args.mode}'"
          + (f", {n_empty} phai lui ve benh an goc" if n_empty else ""))
    avg = sum(len(q.split()) for q in queries.values()) / len(queries)
    print(f"  do dai trung binh {avg:.0f} tu  (benh an goc: "
          f"{sum(len(v.split()) for v in topics.values())/len(topics):.0f} tu)")

    run = bm25.search(args.index, queries, args.k1, args.b, args.depth)
    out = args.out or f"runs/bm25_{args.mode}.dev.txt"
    tag = out.split("/")[-1].rsplit(".", 1)[0]
    run_io.write_run(out, run, tag, depth=args.depth)
    print(f"Da ghi {out}")
    print(f"\nCham diem:\n  PYTHONPATH=. .venv/bin/python -m src.eval.score "
          f"{out} --year {args.year} --vs results/bm25_best.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
