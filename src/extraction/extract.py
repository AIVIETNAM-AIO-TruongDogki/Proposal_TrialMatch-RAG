"""Phase 4 step 3 — run batched extraction, cached.

    python -m src.extraction.extract --year 2021 --model gemini-3.6-flash

Writes data/profiles/{year}.{model}.json. Each record keeps BOTH the raw
`profile` and the grounding-filtered `clean` version plus `dropped` — kept
deliberately, since the drop list shows what kind of thing the model
fabricates, which the hand-audit in step 5 needs.

Batches `--batch-size` patients per call (default 5) to cut request count.
Each returned profile carries its own `index` to match back; a missing/
duplicate/out-of-range index only breaks that one patient, not the batch
(see schema.batch_schema).

Cache key is (year, model, prompt_hash) — a prompt or schema change
invalidates it, so results from two different questions are never compared
as if they were two models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.eval import data
from src.extraction import gemini, schema, verify

PROFILE_DIR = "data/profiles"
BATCH_SIZE = 5


def load_cache(path: str, ph: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        blob = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if blob.get("prompt_hash") != ph:
        print(f"  prompt/schema da doi ({blob.get('prompt_hash')} -> {ph}); bo cache cu")
        return {}
    return blob.get("records", {})


def _save(path: str, ph: str, model: str, recs: dict) -> None:
    json.dump({"prompt_hash": ph, "model": model, "records": recs},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def extract_one(narrative: str, model: str = gemini.MODEL
                ) -> tuple[dict | None, list[dict], dict]:
    """Extract ONE narrative at request time — no topic/year, no disk cache.

    Returns (clean, dropped, meta). `clean` is None if the model returns an
    invalid schema. Reuses `schema.batch_schema(1)` /
    `schema.batch_user_prompt([narrative])` as-is — the batch schema already
    handles any array length, so n=1 needs no schema of its own.
    """
    out, meta = gemini.chat_json(model, schema.BATCH_SYSTEM_PROMPT,
                                 schema.batch_user_prompt([narrative]),
                                 schema.batch_schema(1))
    items = (out or {}).get("profiles") or []
    it = next((x for x in items if isinstance(x, dict) and x.get("index") == 0), None)
    if it is None:
        return None, [], meta
    prof = {k: v for k, v in it.items() if k != "index"}
    if not verify.schema_ok(prof):
        return None, [], meta
    clean, dropped = verify.verify_profile(prof, narrative)
    return clean, dropped, meta


def run(model: str, year: int, limit: int | None = None,
        profile_dir: str = PROFILE_DIR, force: bool = False,
        batch_size: int = BATCH_SIZE) -> int:
    topics = data.load_topics(year)
    if limit:
        topics = dict(list(topics.items())[:limit])

    ph = schema.prompt_hash()
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, f"{year}.{model.replace(':', '_')}.json")
    recs = {} if force else load_cache(path, ph)

    todo = [t for t in topics if t not in recs]
    n_batches = -(-len(todo) // batch_size) if todo else 0
    print(f"{model}: {len(topics)} benh an, {len(recs)} da co cache, "
          f"{len(todo)} can goi ({n_batches} lo x{batch_size})")

    t0 = time.time()
    done = 0
    for bi, start in enumerate(range(0, len(todo), batch_size), 1):
        batch_ids = todo[start:start + batch_size]
        narratives = [topics[tid] for tid in batch_ids]

        try:
            out, meta = gemini.chat_json(
                model, schema.BATCH_SYSTEM_PROMPT,
                schema.batch_user_prompt(narratives),
                schema.batch_schema(len(batch_ids)))
        except gemini.GeminiError as e:
            # Transient failures (quota, network) are NOT written to recs, so
            # the next run retries automatically instead of treating them as
            # permanently done. Unlike the failure branch below (bad JSON /
            # index match), which rarely self-corrects on a retry with the
            # same input, so that one IS cached.
            done += len(batch_ids)
            print(f"  lo {bi}/{n_batches} LOI TAM THOI (se thu lai o lan chay "
                  f"sau): {e}", file=sys.stderr)
            continue

        # Split evenly across the batch; prompt/output tokens stay batch-total
        # below — the system prompt is paid once per batch, not N times.
        per_topic_seconds = round(meta["seconds"] / len(batch_ids), 2)
        items = (out or {}).get("profiles") or []
        by_index: dict[int, dict] = {}
        for it in items:
            idx = it.get("index") if isinstance(it, dict) else None
            if isinstance(idx, int) and 0 <= idx < len(batch_ids) and idx not in by_index:
                by_index[idx] = it
            # missing/duplicate/out-of-range index: skip — that patient falls
            # into the "failed" branch below instead of matching the wrong one.

        for i, tid in enumerate(batch_ids):
            narrative = narratives[i]
            it = by_index.get(i)
            prof = {k: v for k, v in it.items() if k != "index"} if it else None

            rec: dict = {"seconds": per_topic_seconds,
                        "prompt_tokens": meta["prompt_tokens"],
                        "output_tokens": meta["output_tokens"],
                        "batch_size": len(batch_ids),
                        "profile": prof}
            if prof is not None and verify.schema_ok(prof):
                rec["clean"], rec["dropped"] = verify.verify_profile(prof, narrative)
            else:
                # Keep the raw string for debugging: knowing WHAT the model
                # returned when it fails matters more than just that it failed.
                rec["clean"], rec["dropped"] = None, None
                rec["raw"] = meta["raw"][:2000]
            recs[tid] = rec

        done += len(batch_ids)
        el = time.time() - t0
        print(f"  lo {bi}/{n_batches}  {done}/{len(todo)} benh an  {el:.0f}s  "
              f"({el/done:.1f}s/benh an)", flush=True)
        _save(path, ph, model, recs)

    # Unconditional final save: if the last batch failed transiently (no
    # _save() of its own), the file must still reflect recs before printing below.
    _save(path, ph, model, recs)

    n_bad = sum(1 for r in recs.values() if r.get("clean") is None)
    n_drop = sum(len(r["dropped"]) for r in recs.values() if r.get("dropped"))
    print(f"Da ghi {path}")
    print(f"  {len(recs)} ban ghi, {n_bad} hong schema, {n_drop} gia tri bi vut "
          f"vi khong trich dan duoc")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=gemini.MODEL)
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--limit", type=int, default=None, help="chi chay N benh an dau")
    ap.add_argument("--profile-dir", default=PROFILE_DIR)
    ap.add_argument("--force", action="store_true", help="bo qua cache")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="so benh an moi lan goi Gemini (mac dinh 5)")
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! Dang chay tren TAP TEST 2022 — Phase 4 chi lam tren dev.",
              file=sys.stderr)

    if not gemini.KEYS:
        print("Khong co GEMINI_API_KEY_1/2/3 nao trong .env. Xem .env.example.",
              file=sys.stderr)
        return 1

    return run(args.model, args.year, args.limit, args.profile_dir, args.force,
              args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
