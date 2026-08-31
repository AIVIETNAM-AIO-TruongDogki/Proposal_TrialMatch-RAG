"""Phase 8 buoc 2 — chay suy luan eligibility muc tieu chi.

    # smoke test: 1 topic x 2 trial, vai lan goi
    python -m src.reasoning.reason --limit-topics 1 --top-n 2

    # chay that (KHONG chay khi chua co quota — xem canh bao ben duoi)
    python -m src.reasoning.reason --run runs/bm25_best.dev.txt --top-n 20

HAI CHE DO GOI — SPEC YEU CAU DO CA HAI
----------------------------------------
--mode trial     gop CA TRIAL vao mot lan goi (~18 tieu chi).   1.500 lan goi
--mode criterion goi tung tieu chi mot.                        27.045 lan goi

Che do `trial` re hon 18 lan nhung danh cuoc rang model khong lac giua 18 tieu
chi trong mot prompt. Che do `criterion` dat hon nhung moi lan goi chi phai
tra loi dung mot cau hoi. Do ca hai roi hay chon, dung doan.

QUY MO — DOC TRUOC KHI CHAY
----------------------------
75 topic x top-20 x 18,0 tieu chi/trial = 27.045 lan goi (do that tren
runs/bm25_best.dev.txt, khong phai uoc luong). Han ngach free tier do duoc la
20 request/ngay/model. Ke ca duong `trial` 1.500 lan goi cung la 75 NGAY.
`--estimate` in ra so lan goi va dung lai, khong goi gi ca.

CACHE
-----
Khoa (topic_id, nct_id, criterion_idx) + prompt_hash. Loi ha tang KHONG duoc
ghi cache (lan chay sau tu thu lai); dau ra sai dinh dang thi CO ghi, vi chay
lai cung dau vao hiem khi tu sua. Giong het extract.py.
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

# Do bang thuc nghiem: Gemini tra 400 INVALID_ARGUMENT khi mot lan goi mang
# ~40 tieu chi tro len (n=35 con chay, n=40 hong). Khong phai gioi han token —
# la gioi han do PHUC TAP cua response_schema (minItems/maxItems lon + object
# long nhau). 30 la nguong an toan co bien. 10,3% trial trong top-20 co hon 30
# tieu chi (toi da 76), nen chia nho la BAT BUOC, khong phai toi uu.
MAX_CRIT_PER_CALL = 30


def cache_path(year: int, model: str, mode: str, forced: bool,
               out_dir: str = OUT_DIR) -> str:
    f = ".forced" if forced else ""
    return os.path.join(out_dir, f"{year}.{model.replace(':', '_')}.{mode}{f}.json")


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


def save_cache(path: str, ph: str, model: str, mode: str, recs: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump({"prompt_hash": ph, "model": model, "mode": mode, "records": recs},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def plan_work(run: dict, topics: dict, conn, top_n: int,
              limit_topics: int | None = None) -> list[tuple[str, str, list[dict]]]:
    """(topic_id, nct_id, criteria[]) cho tung cap can suy luan."""
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
    """Goi theo trial, chia nho neu trial co qua nhieu tieu chi.

    Khop lai theo `criterion_idx` chu KHONG theo thu tu mang: mot lan khop sai
    se gan quyet dinh cua tieu chi nay cho tieu chi khac — dung dieu invariant
    3 cam. Chia nho khong lam thay doi dieu do: moi mieng van kiem idx cua
    rieng no, va `seen` dung chung ca trial nen trung lap giua cac mieng cung
    bi bat.
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
        # Trial nhieu hon MAX_CRIT_PER_CALL tieu chi can nhieu hon mot lan goi.
        calls = sum(-(-len(c) // args.max_criteria) for _, _, c in work)
    else:
        calls = n_crit
    print(f"{len(work):,} cap (topic,trial), {n_crit:,} tieu chi "
          f"({n_crit/max(len(work),1):.1f}/trial)")
    print(f"Che do '{args.mode}' -> {calls:,} lan goi API")
    if args.estimate:
        print(f"\nHan ngach do duoc: 20 request/ngay/model.")
        print(f"  {calls:,} lan goi = {calls/20:.0f} NGAY tren free tier mot model.")
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
            # TAM THOI — khong ghi cache, lan chay sau tu thu lai.
            print(f"  [{i}/{len(todo)}] {tid}/{nct} LOI TAM THOI: {e}",
                  file=sys.stderr)
            continue

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
