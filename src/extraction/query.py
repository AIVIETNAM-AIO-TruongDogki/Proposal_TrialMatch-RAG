"""Phase 4 buoc 4 — ho so -> truy van BM25, va do lai voi Phase 3.

    python -m src.extraction.query --model gemini-3.6-flash --mode prof

BA BIEN THE
-----------
  prof        chi cac term da trich (conditions + biomarkers + treatments + comorbidities)
  prof_narr   benh an goc + term da trich (noi them)
  hyde        sinh mot mo ta thu nghiem gia dinh tu ho so, truy hoi bang no

HYDE GOI THEO LO VA CO CACHE RA DIA
-----------------------------------
`hyde` la bien the DUY NHAT ton quota LLM o buoc nay. Goi tung topic mot se
can 75 request — vuot xa han muc free tier 20/ngay. gen_hyde_batch() gom
--hyde-batch-size topic moi lan goi (mac dinh 5 -> 15 request) va ghi ket qua
ra data/profiles/hyde.{year}.{model}.json, khoa theo prompt_hash. Chay lai
ablation sau nay khong ton them mot request nao.

Moi mo ta tra ve kem `index` de khop lai DUNG topic — index thieu/trung/ngoai
khoang chi lam hong topic do (lui ve profile_summary), khong bao gio gan nham
mo ta cua benh nhan nay cho benh nhan khac.

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

import prompts
from src.eval import data, run_io
from src.extraction import gemini, schema

BEST_INDEX = "indexes/bm25-critfields"   # cau hinh thang cuoc Phase 3
BEST_K1, BEST_B = 1.8, 1.0

HYDE_BATCH_SIZE = 5
HYDE_CACHE_DIR = "data/profiles"
HYDE_BATCH_SYSTEM = prompts.load("hyde_system") + "\n\n" + prompts.load("hyde_batch_addendum")


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


def profile_summary(profile: dict) -> str:
    """Mo ta ngan gon tu ho so, dung lam input cho HyDE (va lam phuong an lui)."""
    terms = profile_terms(profile)
    desc = "; ".join(terms) or "(no extracted findings)"
    age = (profile.get("age") or {}).get("value")
    sex = (profile.get("sex") or {}).get("value")
    who = f"{age}-year-old {sex}" if age and sex else "patient"
    return f"A {who} with: {desc}"


def _hyde_batch_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
                    "required": ["index", "text"],
                    "additionalProperties": False,
                },
                "minItems": n, "maxItems": n,
            },
        },
        "required": ["descriptions"],
        "additionalProperties": False,
    }


def _hyde_batch_prompt(summaries: list[str]) -> str:
    header = prompts.load("hyde_batch_header").format(n=len(summaries))
    item_tpl = prompts.load("hyde_batch_item")
    items = "\n\n".join(item_tpl.format(index=i, profile_summary=s)
                        for i, s in enumerate(summaries))
    return f"{header}\n\n{items}"


def _hyde_prompt_hash() -> str:
    import hashlib
    blob = (HYDE_BATCH_SYSTEM + prompts.load("hyde_batch_header") +
            prompts.load("hyde_batch_item"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _hyde_cache_path(model: str, year: int, cache_dir: str = HYDE_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"hyde.{year}.{model.replace(':', '_')}.json")


def _load_hyde_cache(path: str, ph: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        blob = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if blob.get("prompt_hash") != ph:
        print(f"  hyde prompt da doi ({blob.get('prompt_hash')} -> {ph}); bo cache cu")
        return {}
    return blob.get("texts", {})


def _save_hyde_cache(path: str, ph: str, texts: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"prompt_hash": ph, "texts": texts},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def gen_hyde_batch(model: str, year: int, profiles: dict[str, dict],
                   batch_size: int = HYDE_BATCH_SIZE,
                   cache_dir: str = HYDE_CACHE_DIR) -> dict[str, str]:
    """Sinh mo ta HyDE cho tat ca topic co ho so, theo lo, co cache ra dia.

    Cung nguyen tac an toan voi extract.py: index thieu/trung/ngoai khoang chi
    lam hong dung topic do (lui ve profile_summary), khong lam sai lech ca lo
    va khong bao gio gan nham mo ta cua benh nhan nay cho benh nhan khac.
    """
    ph = _hyde_prompt_hash()
    path = _hyde_cache_path(model, year, cache_dir)
    texts = _load_hyde_cache(path, ph)

    ids = list(profiles.keys())
    todo = [tid for tid in ids if tid not in texts]
    n_batches = -(-len(todo) // batch_size) if todo else 0
    if todo:
        print(f"hyde: {len(ids)} topic co ho so, {len(texts)} da co cache, "
              f"{len(todo)} can goi ({n_batches} lo x{batch_size})")

    for bi, start in enumerate(range(0, len(todo), batch_size), 1):
        batch_ids = todo[start:start + batch_size]
        summaries = [profile_summary(profiles[tid]) for tid in batch_ids]

        try:
            out, meta = gemini.chat_json(model, HYDE_BATCH_SYSTEM,
                                         _hyde_batch_prompt(summaries),
                                         _hyde_batch_schema(len(batch_ids)))
        except gemini.GeminiError as e:
            # TAM THOI: khong ghi vao texts, lan chay sau tu dong thu lai.
            print(f"  lo {bi}/{n_batches} LOI TAM THOI (se thu lai o lan chay "
                  f"sau): {e}", file=sys.stderr)
            continue

        items = (out or {}).get("descriptions") or []
        by_index: dict[int, dict] = {}
        for it in items:
            idx = it.get("index") if isinstance(it, dict) else None
            if isinstance(idx, int) and 0 <= idx < len(batch_ids) and idx not in by_index:
                by_index[idx] = it

        for i, tid in enumerate(batch_ids):
            it = by_index.get(i)
            text = (it or {}).get("text") if isinstance(it, dict) else None
            texts[tid] = text or summaries[i]

        print(f"  lo {bi}/{n_batches}  {min(start + batch_size, len(todo))}/{len(todo)} topic")
        _save_hyde_cache(path, ph, texts)

    _save_hyde_cache(path, ph, texts)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=gemini.MODEL, help="model da dung o extract")
    # Tach khoi --model: quota Gemini tinh theo TUNG MODEL, nen khi model dung o
    # extract het quota, van sinh duoc HyDE bang model khac. Ca 75 topic phai
    # dung CUNG mot model — tron model giua cac topic bien nhanh `hyde` thanh
    # "HyDE voi mot mo model hon hop", khong tai lap va khong quy duoc chenh lech.
    ap.add_argument("--hyde-model", default=None,
                    help="model sinh mo ta HyDE (mac dinh: giong --model)")
    ap.add_argument("--mode", default="prof", choices=["prof", "prof_narr", "hyde"])
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--index", default=BEST_INDEX)
    ap.add_argument("--k1", type=float, default=BEST_K1)
    ap.add_argument("--b", type=float, default=BEST_B)
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hyde-batch-size", type=int, default=HYDE_BATCH_SIZE)
    args = ap.parse_args()

    if (args.index, args.k1, args.b) != (BEST_INDEX, BEST_K1, BEST_B):
        print("!! Khong dung cau hinh thang cuoc cua Phase 3 — so sanh voi "
              "bm25_best se KHONG hop le (doi hai thu cung luc).", file=sys.stderr)

    from src.retrieval import bm25

    topics = data.load_topics(args.year)
    recs = load_profiles(args.model, args.year)

    hyde_texts: dict[str, str] = {}
    if args.mode == "hyde":
        profiles_ok = {tid: (recs.get(tid) or {}).get("clean") for tid in topics}
        profiles_ok = {tid: p for tid, p in profiles_ok.items() if p}
        hyde_model = args.hyde_model or args.model
        if hyde_model != args.model:
            print(f"HyDE dung model {hyde_model} (ho so trich bang {args.model}).")
        hyde_texts = gen_hyde_batch(hyde_model, args.year, profiles_ok,
                                    args.hyde_batch_size)

        # Thieu mo ta HyDE thi build_query() lui ve BENH AN GOC — tuc la run se
        # tron lan hai che do khac han nhau. Diem so cua no khong do duoc HyDE
        # ma cung khong do duoc baseline; no khong tra loi cau hoi nao ca.
        # Dung lai thay vi ghi ra mot file trong nhu ket qua hop le. Cache da
        # giu phan da sinh, chay lai khi quota hoi phuc se di tiep tu do.
        missing = [t for t in profiles_ok if t not in hyde_texts]
        if missing:
            print(f"\nDUNG LAI: {len(missing)}/{len(profiles_ok)} topic chua co "
                  f"mo ta HyDE (vd {missing[:3]}).\n"
                  f"Ghi run bay gio se tron {len(missing)} benh an goc vao "
                  f"{len(hyde_texts)} truy van HyDE — khong phai ablation HyDE.\n"
                  f"Chay lai lenh nay khi quota hoi phuc; phan da sinh da nam "
                  f"trong cache, khong ton lai request.", file=sys.stderr)
            return 1

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
        hyde = hyde_texts.get(tid) if args.mode == "hyde" else None
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
