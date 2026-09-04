"""Phase 8 step 2 — run per-criterion eligibility reasoning.

    # smoke test
    python -m src.reasoning.reason --limit-topics 1 --top-n 2

    # real run — check --estimate first, quota is limited
    python -m src.reasoning.reason --run runs/bm25_best.dev.txt --top-n 20

Two call modes: `trial` batches all of a trial's criteria into one call
(~1,500 calls); `criterion` calls one at a time (~27,045 calls) — cheaper
per-criterion accuracy at 18x the cost. `--estimate` prints call counts and
exits without calling anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.corpus import store
from src.eval import data, run_io
from src.extraction import gemini
from src.reasoning import schema, verify

OUT_DIR = "data/reasoning"
TOP_N = 20

# Measured empirically: Gemini 400s on ~40+ criteria per call (35 still works,
# 40 fails) — not a token limit but response_schema complexity (large
# minItems/maxItems + nested objects). 30 is a safe margin; 10.3% of top-20
# trials have more than 30 criteria (max 76), so splitting is mandatory.
MAX_CRIT_PER_CALL = 30

# Consecutive failures, not sporadic ones, mean the quota is exhausted. 5 is
# enough: a stray 429 gets picked up by the next key in rotation, so 5
# straight failures across all 3 keys means all 3 projects are out.
MAX_CONSECUTIVE_FAILS = 5


def cache_path(year: int, model: str, mode: str, forced: bool,
               out_dir: str = OUT_DIR) -> str:
    f = ".forced" if forced else ""
    return os.path.join(out_dir, f"{year}.{model.replace(':', '_')}.{mode}{f}.json")


def load_cache(path: str, ph: str) -> dict:
    """Load the cache. A malformed file is never treated as an empty one.

    Swallowing JSONDecodeError and returning {} would make a truncated file
    (killed mid-write, disk full) silently re-call everything on the next run,
    burning a day's quota with no warning. Fail loud instead, and point at `.bak`.
    """
    if not os.path.exists(path):
        return {}
    try:
        blob = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError as e:
        bak = path + ".bak"
        raise SystemExit(
            f"Cache HONG: {path}\n  {e}\n"
            f"  KHONG tu dong chay lai — lam vay se am tham goi lai tu dau.\n"
            f"  Neu co {bak} thi khoi phuc no; neu that su muon bo, xoa file di.")
    if blob.get("prompt_hash") != ph:
        print(f"  prompt/schema da doi ({blob.get('prompt_hash')} -> {ph}); bo cache cu")
        return {}
    return blob.get("records", {})


def save_cache(path: str, ph: str, model: str, mode: str, recs: dict) -> None:
    """Write atomically: temp file + os.replace, keeping the previous file as `.bak`.

    A direct `json.dump` leaves a multi-second window where a kill truncates
    the real file, destroying a day's worth of results. os.replace is atomic
    on the same filesystem — the old file survives intact or the new lands whole.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"prompt_hash": ph, "model": model, "mode": mode,
                   "records": recs}, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())   # flush to disk before the rename
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def plan_work(run: dict, topics: dict, conn, top_n: int,
              limit_topics: int | None = None) -> list[tuple[str, str, list[dict]]]:
    """(topic_id, nct_id, criteria[]) for every pair that needs reasoning."""
    tids = sorted(run)
    if limit_topics:
        tids = tids[:limit_topics]
    work = []
    for tid in tids:
        if tid not in topics:
            continue
        top = [d for d, _ in sorted(run[tid].items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:top_n]]
        for nct in top:
            crit = store.get_criteria(conn, nct)
            if crit:
                work.append((tid, nct, crit))
    return work


def _accept(conn, dec: dict, tid: str, nct: str, idx: int, narrative: str,
            section: str, rejections: dict) -> dict | None:
    ok, why = verify.check(conn, dec, nct, idx, narrative)
    if not ok:
        rejections[why] = rejections.get(why, 0) + 1
        return None
    return {"criterion_idx": idx, "section": section, "label": dec["label"],
            "criterion_quote": dec.get("criterion_quote", ""),
            "patient_evidence": dec.get("patient_evidence", ""),
            "reasoning": (dec.get("reasoning") or "")[:300]}


def run_batch_trial(model: str, tid: str, nct: str, crit: list[dict],
                    narrative: str, conn, forced: bool, rejections: dict,
                    max_per_call: int = MAX_CRIT_PER_CALL
                    ) -> tuple[list[dict], dict]:
    """Call per trial, splitting into multiple calls if it has too many criteria.

    Matches results back by `criterion_idx`, never by array order — a mismatch
    would attribute one criterion's decision to another (invariant 3). Splitting
    doesn't weaken this: each piece still checks its own indices, and `seen` is
    shared across the whole trial so duplicates across pieces are still caught.
    """
    valid_idx = {c["idx"]: c["section"] for c in crit}
    seen: set[int] = set()
    kept: list[dict] = []
    secs, ptok, otok, raw = 0.0, 0, 0, ""

    for s in range(0, len(crit), max_per_call):
        piece = crit[s:s + max_per_call]
        out, meta = gemini.chat_json(
            model, schema.BATCH_SYSTEM,
            schema.batch_user_prompt(narrative, nct, piece),
            schema.batch_schema(len(piece), forced))
        secs += meta["seconds"]
        ptok += meta["prompt_tokens"] or 0
        otok += meta["output_tokens"] or 0
        raw = raw or (meta.get("raw") or "")

        for d in ((out or {}).get("decisions") or []):
            if not isinstance(d, dict):
                continue
            i = d.get("criterion_idx")
            if not isinstance(i, int) or i not in valid_idx or i in seen:
                rejections["criterion_idx_bad"] = rejections.get("criterion_idx_bad", 0) + 1
                continue
            seen.add(i)
            rec = _accept(conn, d, tid, nct, i, narrative, valid_idx[i], rejections)
            if rec:
                kept.append(rec)

    kept.sort(key=lambda r: r["criterion_idx"])
    return kept, {"seconds": round(secs, 2), "prompt_tokens": ptok,
                  "output_tokens": otok, "raw": raw[:2000]}


def run_per_criterion(model: str, tid: str, nct: str, crit: list[dict],
                      narrative: str, conn, forced: bool, rejections: dict
                      ) -> tuple[list[dict], dict]:
    kept, secs, ptok, otok = [], 0.0, 0, 0
    for c in crit:
        out, meta = gemini.chat_json(
            model, schema.SYSTEM,
            schema.user_prompt(narrative, nct, c["section"], c["text"]),
            schema.decision_schema(forced))
        secs += meta["seconds"]
        ptok += meta["prompt_tokens"] or 0
        otok += meta["output_tokens"] or 0
        if isinstance(out, dict):
            rec = _accept(conn, out, tid, nct, c["idx"], narrative,
                          c["section"], rejections)
            if rec:
                kept.append(rec)
        else:
            rejections["json_invalid"] = rejections.get("json_invalid", 0) + 1
    return kept, {"seconds": round(secs, 2), "prompt_tokens": ptok,
                  "output_tokens": otok, "raw": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bm25_best.dev.txt")
    ap.add_argument("--model", default=gemini.MODEL)
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--limit-topics", type=int, default=None,
                    help="chi N topic dau — dung cho smoke test")
    ap.add_argument("--mode", default="trial", choices=["trial", "criterion"])
    ap.add_argument("--forced", action="store_true",
                    help="ablation: bo nhan unverifiable (kiem chung invariant 1)")
    ap.add_argument("--estimate", action="store_true",
                    help="in so lan goi can thiet roi dung, KHONG goi API")
    ap.add_argument("--max-criteria", type=int, default=MAX_CRIT_PER_CALL,
                    help="so tieu chi toi da moi lan goi (Gemini hong o ~40)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--db", default="data/trials.db")
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! TAP TEST 2022 — chi cham MOT LAN o Phase 11.", file=sys.stderr)

    topics = data.load_topics(args.year)
    run = run_io.read_run(args.run)
    conn = store.open_db(args.db)
    work = plan_work(run, topics, conn, args.top_n, args.limit_topics)
    n_crit = sum(len(c) for _, _, c in work)

    if args.mode == "trial":
        # A trial with more than MAX_CRIT_PER_CALL criteria needs more than one call.
        calls = sum(-(-len(c) // args.max_criteria) for _, _, c in work)
    else:
        calls = n_crit
    print(f"{len(work):,} cap (topic,trial), {n_crit:,} tieu chi "
          f"({n_crit/max(len(work),1):.1f}/trial)")
    print(f"Che do '{args.mode}' -> {calls:,} lan goi API")
    if args.estimate:
        # Quota is per model, not a flat number — 20/day was measured on
        # gemini-3.6-flash, not the model in use. And it scales with PROJECT
        # count, not key count: confirmed 2026-09-03 that .env's keys belong
        # to separate projects, so the real cap is nk x per-model quota.
        nk = max(len(gemini.KEYS), 1)
        print(f"\nHan ngach — theo TUNG model, khong suy rong duoc "
              f"({nk} khoa = {nk} du an rieng):")
        for lab, cap in (("gemini-3.6-flash (do bang 429)", 20),
                         ("gemini-3.5-flash-lite (>30 do duoc; bao cao 500)", 500)):
            tong = cap * nk
            print(f"  {lab:48s} {calls/tong:5.1f} ngay  ({tong:,}/ngay)")
        print(f"\nThoi gian API that: ~{calls * 5.1 / 3600:.1f} gio "
              f"(5,1 s/goi do tren smoke test) — phan con lai la CHO han ngach.")
        print("Dung lai (--estimate). Bo co nay de chay that.")
        return 0

    if not gemini.KEYS:
        print("Khong co GEMINI_API_KEY_* trong .env.", file=sys.stderr)
        return 1

    ph = schema.prompt_hash(args.forced, args.mode == "trial")
    path = cache_path(args.year, args.model, args.mode, args.forced, args.out_dir)
    recs = load_cache(path, ph)
    todo = [w for w in work if f"{w[0]}|{w[1]}" not in recs]
    print(f"{len(recs)} cap da co cache, {len(todo)} can goi\n")

    rejections: dict[str, int] = {}
    n_dec = 0
    fails = 0
    t0 = time.time()
    for i, (tid, nct, crit) in enumerate(todo, 1):
        try:
            if args.mode == "trial":
                kept, meta = run_batch_trial(args.model, tid, nct, crit,
                                              topics[tid], conn, args.forced,
                                              rejections, args.max_criteria)
            else:
                kept, meta = run_per_criterion(args.model, tid, nct, crit,
                                                topics[tid], conn, args.forced,
                                                rejections)
        except gemini.GeminiError as e:
            # Transient — not cached, retried automatically on the next run.
            fails += 1
            print(f"  [{i}/{len(todo)}] {tid}/{nct} LOI TAM THOI: {e}",
                  file=sys.stderr)
            # Circuit breaker: once quota is truly exhausted every remaining
            # pair fails identically, but chat_json still retries all 3 keys
            # with up to ~20s sleep each (~60s/pair) — thousands of pairs would
            # waste hours. Stop cleanly on consecutive failures instead; a
            # rerun resumes from cache.
            if fails >= MAX_CONSECUTIVE_FAILS:
                print(f"\n!! {fails} loi LIEN TIEP — nhieu kha nang het han ngach "
                      f"ngay. Dung lai va ghi cache; chay lai lenh nay se tiep tuc "
                      f"tu {len(recs):,} cap da xong.", file=sys.stderr)
                break
            continue
        fails = 0

        recs[f"{tid}|{nct}"] = {"decisions": kept, "n_criteria": len(crit),
                                "seconds": meta["seconds"],
                                "prompt_tokens": meta["prompt_tokens"],
                                "output_tokens": meta["output_tokens"]}
        n_dec += len(kept)
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)} cap  {el:.0f}s  {n_dec:,} quyet dinh giu lai",
                  flush=True)
            save_cache(path, ph, args.model, args.mode, recs)

    save_cache(path, ph, args.model, args.mode, recs)
    total = n_dec + sum(rejections.values())
    print(f"\nDa ghi {path}")
    print("KIEM CHUNG GROUNDING (ty le vi pham la mot ket qua, khong chi la bo loc):")
    print(verify.summarize(rejections, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
