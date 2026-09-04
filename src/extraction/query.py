"""Phase 4 step 4 — profile -> BM25 query, re-scored against Phase 3.

    python -m src.extraction.query --model gemini-3.6-flash --mode prof

Three variants: `prof` (extracted terms only), `prof_narr` (narrative + terms),
`hyde` (a synthetic trial description generated per profile, batched and
cached — the only variant that costs LLM quota). Negated terms stay in the
profile for Phase 8 but are excluded from the query; BM25 can't read negation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.eval import data, run_io
from src.extraction import gemini, schema

BEST_INDEX = "indexes/bm25-critfields"   # Phase 3's winning configuration
BEST_K1, BEST_B = 1.8, 1.0

HYDE_BATCH_SIZE = 5
HYDE_CACHE_DIR = "data/profiles"

_HYDE_SYSTEM = """You write a short, plausible clinical trial description for which the given patient would be a strong candidate. Write 3-5 sentences in the style of a ClinicalTrials.gov brief summary: condition studied, target population, and intervention type. Use standard medical terminology. Do not mention the patient, and do not invent specific trial names or NCT numbers."""

_HYDE_BATCH_ADDENDUM = """You will receive MULTIPLE independent patient summaries in this one request, each labeled with an index number. Write one trial description per patient, completely independently of the others — do not blend details across patients.

Return a JSON object with a single field "descriptions": an array with exactly one entry per patient given, each entry carrying its own "index" field (matching the label below) and a "text" field with that patient's trial description."""

_HYDE_BATCH_HEADER = "Write a short, plausible clinical trial description for each of the {n} independent patients below."

_HYDE_BATCH_ITEM = """--- Patient {index} ---

{profile_summary}"""

HYDE_BATCH_SYSTEM = _HYDE_SYSTEM + "\n\n" + _HYDE_BATCH_ADDENDUM


def profile_terms(profile: dict, include_negated: bool = False) -> list[str]:
    """Terms fed into the query. Negated terms excluded by default — see module docstring."""
    out: list[str] = []
    for f in schema.QUERY_FIELDS:
        for it in profile.get(f) or []:
            if it.get("status") == "negated" and not include_negated:
                continue
            name = (it.get("name") or "").strip()
            if name:
                out.append(name)
    # Dedupe, keep order — order doesn't affect BM25 score but helps log reading.
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
    """Short profile summary, used as HyDE input (and as its fallback)."""
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
    header = _HYDE_BATCH_HEADER.format(n=len(summaries))
    items = "\n\n".join(_HYDE_BATCH_ITEM.format(index=i, profile_summary=s)
                        for i, s in enumerate(summaries))
    return f"{header}\n\n{items}"


def _hyde_prompt_hash() -> str:
    import hashlib
    blob = HYDE_BATCH_SYSTEM + _HYDE_BATCH_HEADER + _HYDE_BATCH_ITEM
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
    """Generate HyDE descriptions for every profiled topic, batched, disk-cached.

    Same safety rule as extract.py: a missing/duplicate/out-of-range index only
    breaks that one topic (falls back to profile_summary), never misattributes
    one patient's description to another.
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
            # Transient — not written to texts, retried automatically next run.
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
    # Separate from --model: Gemini quota is per-model, so HyDE can still run
    # on another model once extraction's model is exhausted. All topics in one
    # run must share a model — mixing mid-run makes `hyde` non-reproducible.
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

        # A missing HyDE description falls back to the raw narrative — a silent
        # mix of two modes that measures neither. Stop instead of writing a run
        # that looks valid; the cache keeps progress, so a retry resumes from it.
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
            # Failed extraction: fall back to the raw narrative instead of
            # dropping the topic — dropping would inflate the average score.
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
