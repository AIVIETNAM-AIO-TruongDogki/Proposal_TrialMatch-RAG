"""Phase 11 step 2 — error taxonomy on the discriminating case.

    python -m src.eval.errors --year 2022 --run runs/elig_strict.test.txt

The case that matters is `qrel == EXCLUDED(1)`: medically relevant, but the
criteria rule the patient out. A trial like that sitting in the top 10 is
exactly the false positive this project exists to remove, so every one of them
is pulled out and asked the same question: *why did the system fail to
disqualify it?*

Four buckets come from specs/11; a fifth is forced by the data:

  flagged      the reasoner DID disqualify it (strict score = 0) — it is in the
               top 10 only because fewer than k reasoned candidates survived.
               A ranking-depth artifact, not a reasoning failure.
  parsing      the excluding rule was never shown to the model: the criteria
               blob carries an exclusion header the segmenter did not turn into
               exclusion criteria (Phase 1 failure).
  extraction   the rule was shown and answered `unverifiable`, but the narrative
               DOES talk about it (content-word overlap) — the fact was in the
               patient text and never made it into the decision.
  reasoning    the rule was shown, the model quoted patient evidence, and still
               answered `satisfied` — it read the right text and drew the wrong
               conclusion.
  ambiguous    everything answered `unverifiable` with nothing in the narrative
               to answer it with. Under invariant 2 this is CORRECT behaviour:
               the exclusion depends on a lab/scan/history the narrative never
               states. It is a dataset limit, not a bug.

The automatic split is a triage, not a verdict: `extraction` vs `reasoning` vs
`ambiguous` rests on a lexical overlap heuristic, and the printout carries the
evidence for each case so the assignment can be checked by hand. The counts are
reported with that caveat attached, never as if they were gold labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter

from src.corpus import store
from src.eval import data, run_io
from src.extraction import gemini
from src.reasoning import aggregate, reason

# Words shared by nearly every criterion and every narrative. Left in, they make
# any pair look related and the overlap signal says nothing.
STOP = {
    "patient", "patients", "subject", "subjects", "study", "trial", "treatment",
    "treated", "therapy", "prior", "history", "disease", "diagnosis", "diagnosed",
    "years", "year", "months", "month", "weeks", "week", "days", "male", "female",
    "clinical", "criteria", "criterion", "inclusion", "exclusion", "known",
    "evidence", "signs", "symptoms", "present", "presents", "presenting",
    "receiving", "received", "current", "currently", "within", "prior", "must",
    "without", "requiring", "any", "other", "including", "include", "included",
    "unable", "willing", "able", "informed", "consent", "participation",
    "medical", "condition", "conditions", "normal", "abnormal", "test", "tests",
    "results", "level", "levels", "based", "with", "have", "has", "had",
}

_WORD = re.compile(r"[a-z][a-z0-9\-]{4,}")


def content(text: str) -> set[str]:
    """Content words worth matching on: >=5 chars, not clinical filler."""
    return {w for w in _WORD.findall((text or "").lower()) if w not in STOP}


def overlap(quote: str, narrative: str) -> list[str]:
    return sorted(content(quote) & content(narrative))


def visible_sections(conn: sqlite3.Connection, nct: str) -> dict:
    """What the segmenter actually handed the reasoner for this trial."""
    trial = store.get_trial(conn, nct) or {}
    crit = store.get_criteria(conn, nct)
    secs = Counter(c["section"] for c in crit)
    raw = trial.get("criteria_raw") or ""
    return {
        "n_criteria": len(crit),
        "parse_method": trial.get("parse_method"),
        "n_exclusion": secs.get("exclusion", 0),
        "n_inclusion": secs.get("inclusion", 0),
        "n_unknown": secs.get("unknown", 0),
        "raw_has_exclusion_header": bool(re.search(r"exclusion", raw, re.I)),
    }


def classify(dec: list[dict] | None, vis: dict, narrative: str) -> tuple[str, dict]:
    """Triage one contaminating trial. Returns (bucket, evidence)."""
    if dec is None:
        return "unreasoned", {"note": "ngoai top-N duoc suy luan"}

    if aggregate.trial_score(dec, "strict") == aggregate.DISQUALIFIED:
        vio = [d for d in dec if d["label"] == "violated"]
        return "flagged", {"violated": [d["criterion_quote"][:120] for d in vio][:3]}

    # Did the model ever see an exclusion rule for this trial? `effective_section`
    # is what aggregation uses, so ask the same question aggregation asks.
    n_excl_seen = sum(1 for d in dec if aggregate.effective_section(d) == "exclusion")
    if n_excl_seen == 0 and vis["raw_has_exclusion_header"]:
        return "parsing", {"n_criteria": vis["n_criteria"],
                           "parse_method": vis["parse_method"],
                           "note": "blob co header Exclusion nhung khong tieu chi nao mang nhan exclusion"}
    if vis["n_criteria"] <= 2:
        return "parsing", {"n_criteria": vis["n_criteria"],
                           "parse_method": vis["parse_method"],
                           "note": "phan doan suy bien"}

    # Rule was visible. Where did it go wrong?
    unv_hits = []
    for d in dec:
        if d["label"] != "unverifiable":
            continue
        sh = overlap(d.get("criterion_quote", ""), narrative)
        if len(sh) >= 2:
            unv_hits.append({"quote": d["criterion_quote"][:120], "shared": sh[:6],
                             "section": aggregate.effective_section(d)})
    if unv_hits:
        return "extraction", {"unverifiable_but_narrative_says": unv_hits[:3]}

    sat_excl = [d for d in dec
                if d["label"] == "satisfied"
                and aggregate.effective_section(d) == "exclusion"
                and (d.get("patient_evidence") or "").strip()]
    if sat_excl:
        return "reasoning", {"satisfied_exclusion_with_evidence": [
            {"quote": d["criterion_quote"][:120],
             "evidence": d["patient_evidence"][:120]} for d in sat_excl[:3]]}

    n = len(dec)
    unv = sum(1 for d in dec if d["label"] == "unverifiable")
    return "ambiguous", {"n_criteria": n, "unverifiable": unv,
                         "abstention": round(unv / n, 3) if n else 0.0}


LABELS = ["flagged", "parsing", "extraction", "reasoning", "ambiguous", "unreasoned"]

BLURB = {
    "flagged": "reasoner DA loai — ket top-10 vi thieu ung vien sach",
    "parsing": "tieu chi loai tru khong den duoc mo hinh (loi Phase 1)",
    "extraction": "benh an CO noi, quyet dinh van unverifiable",
    "reasoning": "doc dung bang chung, ket luan sai",
    "ambiguous": "benh an that su khong noi — abstain la DUNG",
    "unreasoned": "ngoai top-N duoc suy luan",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=data.TEST_YEAR, choices=[2021, 2022])
    ap.add_argument("--run", default="runs/elig_strict.test.txt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="trial")
    ap.add_argument("--db", default="data/trials.db")
    ap.add_argument("--out", default="results/_error_taxonomy.json")
    ap.add_argument("--show", type=int, default=0, help="in n truong hop dau moi nhom")
    args = ap.parse_args()

    model = args.model or gemini.MODEL
    cache = json.load(open(reason.cache_path(args.year, model, args.mode, False),
                           encoding="utf-8"))["records"]
    dec_by = {tuple(k.split("|", 1)): v["decisions"] for k, v in cache.items()}

    qrels = data.load_qrels(args.year)
    topics = data.load_topics(args.year)
    run = run_io.read_run(args.run)
    conn = store.open_db(args.db)

    cases, per_topic = [], Counter()
    for tid, docs in run.items():
        rel = qrels.get(tid, {})
        top = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))[:args.k]
        for rank, (nct, _) in enumerate(top, 1):
            if rel.get(nct) != data.EXCLUDED:
                continue
            dec = dec_by.get((tid, nct))
            vis = visible_sections(conn, nct)
            bucket, ev = classify(dec, vis, topics.get(tid, ""))
            per_topic[tid] += 1
            cases.append({"topic": tid, "nct": nct, "rank": rank,
                          "bucket": bucket, "visible": vis, "evidence": ev})

    if not cases:
        print("Khong co trial qrel==1 nao trong top-%d." % args.k)
        return 0

    counts = Counter(c["bucket"] for c in cases)
    n = len(cases)
    print(f"PHAN LOAI LOI — {args.run}, qrels {args.year}, top-{args.k}")
    print(f"{n} trial 'lien quan nhung bi loai' (qrel==1) con nam trong top-{args.k}, "
          f"tren {len(per_topic)}/{len(run)} benh an\n")
    w = max(len(x) for x in LABELS)
    for b in LABELS:
        c = counts.get(b, 0)
        if not c:
            continue
        print(f"  {b:{w}s} {c:4d}  {c/n*100:5.1f}%   {BLURB[b]}")
    print("\nLuu y: 'flagged' KHONG phai loi suy luan, va 'ambiguous' la hanh vi DUNG")
    print("theo bat bien 2. Chi 'parsing' + 'extraction' + 'reasoning' moi la loi that:")
    real = sum(counts.get(b, 0) for b in ("parsing", "extraction", "reasoning"))
    print(f"  {real}/{n} = {real/n*100:.1f}% so truong hop o nhiem la loi sua duoc.")
    print("\nRanh gioi extraction/reasoning/ambiguous dua tren trung tu noi dung giua")
    print("trich dan tieu chi va benh an — la phan loai so bo, can doc tay de xac nhan.")

    if args.show:
        for b in LABELS:
            sel = [c for c in cases if c["bucket"] == b][:args.show]
            if not sel:
                continue
            print(f"\n--- {b} ({BLURB[b]}) ---")
            for c in sel:
                print(f"  {c['topic']} {c['nct']} rank {c['rank']}")
                print("   ", json.dumps(c["evidence"], ensure_ascii=False)[:400])

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"run": args.run, "year": args.year, "k": args.k,
                   "counts": dict(counts), "n_cases": n,
                   "n_topics_affected": len(per_topic), "cases": cases},
                  fh, indent=2, ensure_ascii=False)
    print(f"\nDa ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
