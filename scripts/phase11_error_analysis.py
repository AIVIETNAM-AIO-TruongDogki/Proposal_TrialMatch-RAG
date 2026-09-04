"""Phase 11 step 2 — why each contaminated top-10 trial escaped disqualification.

    PYTHONPATH=. .venv/bin/python scripts/phase11_error_analysis.py

Takes every trial qrels mark EXCLUDED (medically relevant, patient disqualified)
that rung 5 still ranked in the top 10, joins it with the criterion-level
decisions already cached from Phase 8, and buckets it by the mechanism that let
it through. Read-only: no API calls, no re-scoring. Writes
results/_error_taxonomy.2022.json.

Every one of these trials carries zero `violated` labels by construction — under
the `strict` rule one violation disqualifies — so the question is always what the
model cleared, and on what evidence.
"""

from __future__ import annotations

import collections
import json
import re

from src.corpus import store
from src.eval import data, run_io
from src.reasoning import aggregate

RUN = "runs/elig_strict.test.txt"
CACHE = "data/reasoning/2022.gemini-3.5-flash-lite.trial.json"
OUT = "results/_error_taxonomy.2022.json"

# A sweeping negative about the patient as a whole. Real information, but it
# cannot settle a criterion defined by a lab value, a date window or a procedure.
GLOBAL_NEG = re.compile(
    r"no (other |known |significant )*(medical (problems|conditions|history)|"
    r"other medical|history of any)|medical history is (not significant|unremarkable)|"
    r"otherwise healthy|prior medical condition is unremarkable|"
    r"no significant medical history|takes no medications|"
    r"has not received any treatment|is unremarkable|no known medical problems|"
    r"does not (smoke|use)|no other medical", re.I)

# Criteria a stated demographic legitimately settles (a man is not pregnant).
FEMALE_ONLY = re.compile(r"pregnan|breast.?feed|lactating|postmenopausal|menarche|"
                         r"vaginal bleed|endometri|uterus|women with|female (patients|subject)", re.I)
MALE_SUBJ = re.compile(r"\b(man|male|boy|his)\b", re.I)

STOP = set("""the a an of or and to in for with without any all other than more less at
least most within prior previous current currently including include such as who are is
was be been have has had not no non will may must should can subject subject's subjects
patient patients study trial screening baseline visit day days week weeks month months
year years old age aged during from this that these those their there if by on per e g
i e etc""".split())

BUCKET_DOC = {
    "A_ranking_artifact":
        "Reasoner DID disqualify the trial (strict score 0.0); rerank_by_eligibility still "
        "ranks every reasoned trial above all unreasoned ones, so it stays in the top 10. "
        "Not a reasoning error.",
    "B1_irrelevant_quote":
        "An exclusion criterion was cleared (satisfied) citing a verbatim span with no "
        "content-word overlap with the criterion. The quote is real, so the grounding check "
        "passes; it just does not support the claim.",
    "B2_global_negation_overreach":
        "An exclusion criterion was cleared citing a sweeping negative about the patient. "
        "Real information, but it cannot settle criteria defined by a lab value, a date "
        "window, or a specific procedure.",
    "C_unverifiable_dominated":
        "No exclusion was wrongly cleared; the narrative is simply silent and >=60% of "
        "decisions are unverifiable. Under the strict rule silence never disqualifies. A "
        "consequence of design invariant 2, not a model error.",
    "D_other":
        "Residual — no unsupported clearance and not unverifiable-dominated.",
}


def content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (s or "").lower()) if w not in STOP}


def main() -> int:
    qrels = data.load_qrels(2022, data.RAWDATA)
    topics = data.load_topics(2022, data.RAWDATA)
    run = run_io.read_run(RUN)
    recs = json.load(open(CACHE, encoding="utf-8"))["records"]
    conn = store.open_db()

    rows = []
    for tid in sorted(qrels):
        top = sorted(run.get(tid, {}), key=lambda d: -run[tid][d])[:10]
        for rank, nct in enumerate(top, 1):
            if qrels[tid].get(nct) != data.EXCLUDED:
                continue
            decs = recs.get(f"{tid}|{nct}", {}).get("decisions", [])
            if aggregate.trial_score(decs, rule="strict") == aggregate.DISQUALIFIED:
                rows.append({"topic": tid, "nct": nct, "rank": rank,
                             "bucket": "A_ranking_artifact", "unsupported": 0, "detail": ""})
                continue

            by_idx = {d["criterion_idx"]: d for d in decs}
            male = bool(MALE_SUBJ.search(topics[tid]))
            n_glob = n_zero = n_demo = 0
            n_unv = n_dec = 0
            examples = []
            for c in store.get_criteria(conn, nct):
                d = by_idx.get(c["idx"])
                if not d:
                    continue                      # dropped by the grounding check
                n_dec += 1
                n_unv += d["label"] == "unverifiable"
                if c["section"] != "exclusion" or d["label"] != "satisfied":
                    continue
                ev, txt = d.get("patient_evidence", ""), c["text"]
                if FEMALE_ONLY.search(txt) and male:
                    n_demo += 1                   # correct inference, not an error
                elif GLOBAL_NEG.search(ev):
                    n_glob += 1
                    examples.append(f"[{c['idx']}] {txt[:60]} <= \"{ev[:50]}\"")
                elif not (content(ev) & content(txt)):
                    n_zero += 1
                    examples.append(f"[{c['idx']}] {txt[:60]} <= \"{ev[:50]}\"")

            unsupported = n_glob + n_zero
            if unsupported == 0:
                bucket = ("C_unverifiable_dominated"
                          if n_unv / max(n_dec, 1) >= 0.6 else "D_other")
            elif n_glob >= n_zero:
                bucket = "B2_global_negation_overreach"
            else:
                bucket = "B1_irrelevant_quote"

            rows.append({"topic": tid, "nct": nct, "rank": rank, "bucket": bucket,
                         "unsupported": unsupported, "n_global_neg": n_glob,
                         "n_zero_overlap": n_zero, "n_demographic_ok": n_demo,
                         "detail": " | ".join(examples[:3])})

    cnt = collections.Counter(r["bucket"] for r in rows)
    print(f"{len(rows)} ca nhiem top-10 (tap test 2022), "
          f"{len({r['topic'] for r in rows})} benh an\n")
    for b, n in cnt.most_common():
        print(f"  {b:32s} {n:3d}  ({n / len(rows) * 100:4.1f}%)")
    print(f"\n  Tieu chi exclusion bi 'clear' khong co co so: "
          f"{sum(r['unsupported'] for r in rows)}")
    print(f"  Suy luan nhan khau hoc dung (khong tinh loi): "
          f"{sum(r.get('n_demographic_ok', 0) for r in rows)}")

    json.dump({
        "year": 2022, "run": RUN, "rung": "rung5_eligibility",
        "definition": "Trials qrels label EXCLUDED that rung 5 still ranked in the top 10.",
        "n_cases": len(rows), "n_topics_affected": len({r["topic"] for r in rows}),
        "buckets": BUCKET_DOC, "counts": dict(cnt.most_common()),
        "n_unsupported_clearances": sum(r["unsupported"] for r in rows),
        "n_correct_demographic_inferences": sum(r.get("n_demographic_ok", 0) for r in rows),
        "limitations": [
            "Bucket assignment is rule-based (content-word overlap + negation regex), "
            "validated by hand against cases spanning every bucket.",
            "The demographic exemption only covers male subjects, so a few legitimate "
            "age-based clearances (a 3-year-old cannot have had menarche) count as unsupported.",
        ],
        "cases": rows,
    }, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nDa ghi: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
