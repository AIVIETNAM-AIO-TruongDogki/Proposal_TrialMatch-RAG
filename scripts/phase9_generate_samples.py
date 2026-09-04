"""Phase 9 deliverable — sample polished outputs across 10 dev topics.

    PYTHONPATH=. .venv/bin/python scripts/phase9_generate_samples.py --limit 20

Reads the Phase 8 decisions already cached for dev 2021, renders each trial with
`render.trial_card()`, then runs `polish.polish()` over the card. Only the polish
step costs API calls — no trial is re-reasoned.

Writes results/_phase9_samples.dev.json, which is what
`scripts/phase9_citation_check.py` audits for specs/09's exit criterion.
"""

from __future__ import annotations

import argparse
import json
import time

from src.corpus import store
from src.eval import data
from src.generation import polish, render
from src.extraction import gemini
from src.reasoning import aggregate, score as rscore

CACHE = "data/reasoning/2021.gemini-3.5-flash-lite.trial.json"
OUT = "results/_phase9_samples.dev.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="so output (exit criterion doi 20)")
    ap.add_argument("--topics", type=int, default=10, help="trai deu tren N benh an")
    ap.add_argument("--model", default=gemini.MODEL)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()

    dec, _ = rscore.load_decisions(CACHE)
    conn = store.open_db()

    # Spread the sample across topics instead of taking the first N pairs, so the
    # audit sees different narratives rather than one patient's whole shortlist.
    by_topic: dict[str, list] = {}
    for (tid, nct), ds in sorted(dec.items()):
        if ds:
            by_topic.setdefault(tid, []).append((nct, ds))
    topics = sorted(by_topic)[:args.topics]
    per = max(1, -(-args.limit // max(len(topics), 1)))
    work = [(t, nct, ds) for t in topics for nct, ds in by_topic[t][:per]][:args.limit]

    print(f"{len(work)} output tren {len({t for t, _, _ in work})} benh an "
          f"-> {len(work)} lenh goi polish")
    if args.estimate:
        print("Dung lai (--estimate).")
        return 0

    out, t0 = [], time.time()
    for i, (tid, nct, ds) in enumerate(work, 1):
        trial = store.get_trial(conn, nct)
        if trial is None:
            continue
        card = render.trial_card(trial, ds, aggregate.trial_score(ds, "strict"))
        try:
            prose, meta = polish.polish(card, args.model)
        except gemini.GeminiError as e:
            print(f"  [{i}/{len(work)}] {tid}/{nct} LOI: {e}")
            continue
        if prose is None:
            print(f"  [{i}/{len(work)}] {tid}/{nct} JSON khong hop le")
            continue
        out.append({"topic": tid, "nct": nct, "card": card, "prose": prose,
                    "seconds": meta.get("seconds")})
        print(f"  [{i}/{len(work)}] {tid}/{nct}  {len(prose.get('claims', []))} claim")

    json.dump({"model": args.model, "n": len(out),
               "elapsed_seconds": round(time.time() - t0, 1), "samples": out},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nDa ghi {OUT} ({len(out)} output, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
