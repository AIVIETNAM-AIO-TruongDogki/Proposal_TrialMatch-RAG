"""Phase 4 step 2 — mechanical verification. The most important guardrail here.

A locally-run 3-8B model fabricates. That can't be prevented, but it CAN be
detected and dropped — and the drop rate is itself a reportable number. Every
extracted value carries `evidence`, which must be a verbatim substring of the
narrative, matched after whitespace/case normalization (same approach and
philosophy as store.verify_quote()).

Free gold labels: age/sex are regex-extracted from the narrative's first 220
characters as a check (measured on the dev set: age 75/75, sex 74/75).

THIS CHECK DOES NOT CATCH EVERYTHING — read before trusting the number.
Substring verification confirms the EVIDENCE is real; it does NOT confirm
`name` is actually inferable from it. A model can quote correctly and still
attach the wrong label to the quote (diagnosing instead of extracting, which
invariant 2 forbids) — mechanical grounding lets that through. `grounding` is
a NECESSARY condition, not a SUFFICIENT one; the hand-audit in step 5 exists
precisely because of this gap and can't be automated away.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from src.eval import data
from src.extraction.schema import LIST_FIELDS, SCALAR_FIELDS

PROFILE_DIR = "data/profiles"

# --- Regex gold labels (measured: age 75/75, sex 74/75 on dev) --------------

_UNIT = r"(day|week|month|year)"
_AGE_PATS = [
    re.compile(rf"\b(\d{{1,3}})\s*[-]?\s*{_UNIT}s?\s*[-]?\s*old", re.I),
    re.compile(rf"\b(\d{{1,3}})\s*{_UNIT}s?\s+(?=man|woman|male|female)", re.I),
    re.compile(r"\b(\d{1,3})\s*[-]?\s*(?:y/o|yo)\b", re.I),
    re.compile(r"\b(\d{1,3})\s*[MF]\b"),
]
_SEX_AB = re.compile(r"\b\d{1,3}\s*[-]?\s*(?:y/?o|years?\s*[-]?\s*old)?\s*([MF])\b")
_SEX_M = re.compile(r"\b(?:man|male|gentleman|boy)\b", re.I)
_SEX_F = re.compile(r"\b(?:woman|female|lady|girl)\b", re.I)
_PRO_M = re.compile(r"\b(?:he|his|him)\b", re.I)
_PRO_F = re.compile(r"\b(?:she|her|hers)\b", re.I)

# The narrative does state sex, just outside gold_age_sex()'s 220-char
# window. Kept as a trap because the model answers correctly on wrong grounds
# ("Daughter" doesn't prove the patient is female).
FABRICATION_TRAPS = {"2021_14": "sex"}

# Negation cues — used to check whether the model catches `negated`.
NEG_CUES = re.compile(
    r"\b(?:no\s+(?:history|evidence|signs?|prior)|denies|without|"
    r"negative\s+for|ruled\s+out|not\s+(?:on|taking)|free\s+of|absence\s+of)\b",
    re.I)


def gold_age_sex(text: str) -> tuple[int | None, str | None, str | None]:
    """(age, unit, sex) inferred by regex. None = narrative doesn't say."""
    head = text[:220]
    age = unit = None
    for p in _AGE_PATS:
        m = p.search(head)
        if m:
            age = int(m.group(1))
            unit = (m.group(2).lower() + "s") if (m.lastindex or 0) >= 2 else "years"
            break

    sex = None
    m = _SEX_AB.search(head)
    if m:
        sex = "male" if m.group(1) == "M" else "female"
    else:
        fm, ff = _SEX_M.search(head), _SEX_F.search(head)
        if fm or ff:
            sex = ("male" if (ff is None or (fm and fm.start() < ff.start()))
                   else "female")
        else:
            pm, pf = _PRO_M.search(head), _PRO_F.search(head)
            if pm or pf:
                sex = ("male" if (pf is None or (pm and pm.start() < pf.start()))
                       else "female")
    return age, unit, sex


# --- Kiem chung -------------------------------------------------------------

def norm(s: str) -> str:
    """Normalize exactly like store.verify_quote(): collapse whitespace, lowercase.

    The narrative can wrap mid-sentence while the model returns one line; a
    raw comparison would falsely reject an otherwise-correct quote.
    """
    return " ".join(s.split()).lower()


# Max length for a quote to still count as "pointing at somewhere specific".
# Narratives average 135 words; a 60-word evidence span pinpoints nothing.
MAX_EVIDENCE_WORDS = 30


def grounded(evidence: str, narrative: str) -> bool:
    e = norm(evidence)
    return bool(e) and e in norm(narrative)


def localized(evidence: str) -> bool:
    """Is the quote short enough to actually count as evidence?

    `grounded` alone is gameable: quoting the whole narrative for every field
    scores 100% grounding while pointing at nothing. Observed on a 3B model.
    Reported as a separate column rather than folded into `grounded`.
    """
    return len(evidence.split()) <= MAX_EVIDENCE_WORDS


def schema_ok(profile: dict) -> bool:
    """Hand-rolled check instead of pulling in the `jsonschema` dependency.

    Checks only what's semantically load-bearing: required lists exist, enums
    are valid, and every item carries evidence.
    """
    if not isinstance(profile, dict):
        return False
    for f in LIST_FIELDS:
        v = profile.get(f)
        if not isinstance(v, list):
            return False
        for it in v:
            if not isinstance(it, dict):
                return False
            if it.get("status") not in ("present", "negated"):
                return False
            if not isinstance(it.get("name"), str) or not isinstance(it.get("evidence"), str):
                return False
    if "age" in profile:
        a = profile["age"]
        if not isinstance(a, dict) or not isinstance(a.get("value"), (int, float)):
            return False
        if not isinstance(a.get("evidence"), str):
            return False
    if "sex" in profile:
        s = profile["sex"]
        if not isinstance(s, dict) or s.get("value") not in ("male", "female"):
            return False
        if not isinstance(s.get("evidence"), str):
            return False
    return True


def verify_profile(profile: dict, narrative: str) -> tuple[dict, list[dict]]:
    """Drop every field whose evidence isn't a real substring.

    Returns (clean_profile, dropped_list). The dropped list is written to disk
    so the step-5 hand-audit can see WHAT the model fabricated, not just how much.
    """
    clean: dict = {}
    dropped: list[dict] = []

    for f in SCALAR_FIELDS:
        v = profile.get(f)
        if not isinstance(v, dict):
            continue
        if grounded(v.get("evidence", ""), narrative):
            clean[f] = v
        else:
            dropped.append({"field": f, "value": v.get("value"),
                            "evidence": v.get("evidence", "")})

    for f in LIST_FIELDS:
        keep = []
        for it in profile.get(f) or []:
            if not isinstance(it, dict):
                continue
            if grounded(it.get("evidence", ""), narrative):
                keep.append(it)
            else:
                dropped.append({"field": f, "value": it.get("name"),
                                "evidence": it.get("evidence", "")})
        clean[f] = keep

    return clean, dropped


def n_values(profile: dict) -> int:
    return (sum(1 for f in SCALAR_FIELDS if f in profile)
            + sum(len(profile.get(f) or []) for f in LIST_FIELDS))


# --- Cham diem mot model ----------------------------------------------------

def score_model(model: str, year: int, topics: dict[str, str],
                profile_dir: str = PROFILE_DIR) -> dict | None:
    path = os.path.join(profile_dir, f"{year}.{model.replace(':', '_')}.json")
    if not os.path.exists(path):
        return None
    recs = json.load(open(path, encoding="utf-8"))["records"]

    n = len(recs)
    valid = tot_vals = ok_vals = loc_vals = 0
    age_hit = age_tot = sex_hit = sex_tot = 0
    fabricated = 0
    neg_topics = neg_hit = 0
    secs: list[float] = []

    for tid, rec in recs.items():
        narrative = topics[tid]
        secs.append(rec.get("seconds", 0.0))
        prof = rec.get("profile")

        if prof is None or not schema_ok(prof):
            continue
        valid += 1

        # Grounding is measured on the RAW profile, before filtering.
        raw_n = n_values(prof)
        clean, dropped = verify_profile(prof, narrative)
        tot_vals += raw_n
        ok_vals += raw_n - len(dropped)
        loc_vals += sum(
            1 for f in SCALAR_FIELDS if f in clean and localized(clean[f]["evidence"])
        ) + sum(
            1 for f in LIST_FIELDS for it in (clean.get(f) or [])
            if localized(it["evidence"]))

        # Age/sex vs. gold — measured on the FILTERED profile, since that's
        # what the system actually uses.
        g_age, g_unit, g_sex = gold_age_sex(narrative)
        if g_age is not None:
            age_tot += 1
            a = clean.get("age") or {}
            if a.get("value") == g_age and (a.get("unit") or "years") == g_unit:
                age_hit += 1
        if g_sex is not None:
            sex_tot += 1
            if (clean.get("sex") or {}).get("value") == g_sex:
                sex_hit += 1

        # Fabrication trap: narrative doesn't say it, model filled it in anyway.
        trap = FABRICATION_TRAPS.get(tid)
        if trap and trap in clean:
            fabricated += 1

        # Did it catch the negation?
        if NEG_CUES.search(narrative):
            neg_topics += 1
            if any(it.get("status") == "negated"
                   for f in LIST_FIELDS for it in (clean.get(f) or [])):
                neg_hit += 1

    pct = lambda a, b: (a / b * 100) if b else float("nan")
    return {
        "model": model,
        "n": n,
        "schema_valid": pct(valid, n),
        "grounding": pct(ok_vals, tot_vals),
        "localized": pct(loc_vals, ok_vals),
        "age_acc": pct(age_hit, age_tot),
        "sex_acc": pct(sex_hit, sex_tot),
        "coverage": (ok_vals / valid) if valid else 0.0,
        "neg_recall": pct(neg_hit, neg_topics),
        "fabricated": fabricated,
        "sec_per_call": (sum(secs) / len(secs)) if secs else 0.0,
    }


# --- CLI --------------------------------------------------------------------

# Phase 8 call count for the dev set, measured on runs/bm25_best.dev.txt:
# 75 topics x top-20 trials x 18.0 criteria/trial.
PHASE8_CALLS = 27045


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=data.DEV_YEAR)
    ap.add_argument("--profile-dir", default=PROFILE_DIR)
    args = ap.parse_args()

    topics = data.load_topics(args.year)
    if not os.path.isdir(args.profile_dir):
        print(f"Chua co {args.profile_dir}. Chay src.extraction.extract truoc.",
              file=sys.stderr)
        return 1

    # Model names can contain dots (`gemini-3.6-flash`), so strip by fixed
    # prefix/suffix rather than split(".") — split would cut mid-name.
    prefix, suffix = f"{args.year}.", ".json"
    models = sorted({f[len(prefix):-len(suffix)].replace("_", ":", 1)
                     for f in os.listdir(args.profile_dir)
                     if f.startswith(prefix) and f.endswith(suffix)})
    rows = [r for m in models if (r := score_model(m, args.year, topics, args.profile_dir))]
    if not rows:
        print("Khong co ket qua nao.", file=sys.stderr)
        return 1

    hdr = (f"{'model':24s} {'schema':>7s} {'ground':>7s} {'local':>7s} "
           f"{'age':>6s} {'sex':>6s} {'cover':>6s} {'neg':>6s} {'bia':>4s} "
           f"{'s/goi':>7s} {'P8':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["grounding"]):
        p8 = r["sec_per_call"] * PHASE8_CALLS / 3600
        print(f"{r['model']:24s} {r['schema_valid']:6.1f}% {r['grounding']:6.1f}% "
              f"{r['localized']:6.1f}% "
              f"{r['age_acc']:5.1f}% {r['sex_acc']:5.1f}% {r['coverage']:6.1f} "
              f"{r['neg_recall']:5.1f}% {r['fabricated']:4d} "
              f"{r['sec_per_call']:6.2f}s {p8:7.1f}h")

    print(f"\nP8 = du phong gio GPU cho Phase 8 tren tap dev ({PHASE8_CALLS:,} lan goi,")
    print("     moi tieu chi mot lan goi). Gop ca trial vao mot lan goi thi chia ~18.")
    print("bia  = so benh an KHONG noi thong tin do ma model van dien "
          f"(bay: {', '.join(FABRICATION_TRAPS)}).")

    out = os.path.join("results", f"_extract_bench.{args.year}.json")
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nDa ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
