"""Phase 9 exit criterion — audit the polished outputs mechanically.

    PYTHONPATH=. .venv/bin/python scripts/phase9_citation_check.py

specs/09 asks for an automated check on a 20-output sample: every claim traces
to a criterion index, and no factual claim appears that is absent from the
structured input. Four checks, each a hard pass/fail:

  C1  every claim's criterion_idx exists in that trial's structured input
  C2  a claim's stated status matches the label the reasoning stage assigned
  C3  every number in the prose appears in the structured input — the strongest
      mechanically checkable proxy for fabrication, since invented doses, dates
      and lab values are what an ungrounded generator adds
  C4  invariant 4: no bare "eligible"/"ineligible" verdict, no enrolment advice

C3 is a proxy, not proof: prose can still misstate something that carries no
number. What it does catch is the failure mode that matters here, and it costs
nothing to run on every output.
"""

from __future__ import annotations

import json
import re
import sys

SAMPLES = "results/_phase9_samples.dev.json"
OUT = "results/_phase9_citation_audit.dev.json"

# "potentially eligible" / "likely ineligible" are the sanctioned forms; a bare
# verdict is not. Checked on the summary, where a verdict would actually land.
BARE_VERDICT = re.compile(
    r"(?<!potentially )(?<!likely )\b(in)?eligible\b(?!.{0,40}clinician review)", re.I)
ENROL_ADVICE = re.compile(r"\b(should|recommend\w*|advise\w*)\b[^.]{0,40}\benrol", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(s: str) -> set[str]:
    return {n.replace(",", ".").rstrip(".0").rstrip(".") or "0"
            for n in NUM.findall(s or "")}


def audit_one(s: dict) -> list[str]:
    fails = []
    rows = {r["criterion_idx"]: r for r in s["card"]["criteria"]}
    prose = s["prose"]
    claims = prose.get("claims", [])

    # Counts are part of the structured input, so a report may quote them —
    # "9 of 71 criteria verified" must not read as a fabricated figure.
    src_text = " ".join(
        [r.get("criterion_quote", "") + " " + r.get("patient_evidence", "") for r in rows.values()]
        + [str(s["card"].get(k, "")) for k in
           ("n_criteria", "n_satisfied", "n_violated", "n_unverifiable")]
        + [str(s["card"].get("n_satisfied", 0) + s["card"].get("n_violated", 0)),
           str(prose.get("n_unknown_collapsed", ""))])
    src_nums = numbers(src_text)

    for c in claims:
        idx = c.get("criterion_idx")
        if idx not in rows:                                              # C1
            fails.append(f"C1 claim tro toi criterion_idx {idx} khong ton tai")
            continue
        label = rows[idx]["label"]                                       # C2
        txt = (c.get("text") or "").lower()
        said = {"satisfied": "satisf" in txt or "meets" in txt,
                "violated": "violat" in txt or "disqualif" in txt or "excludes" in txt,
                "unverifiable": "cannot be determined" in txt or "not stated" in txt
                                or "unverifiable" in txt or "no information" in txt}
        if any(said.values()) and not said[label]:
            fails.append(f"C2 [{idx}] nhan la '{label}' nhung van ban noi khac")
        extra = numbers(c.get("text", "")) - src_nums                    # C3
        if extra:
            fails.append(f"C3 [{idx}] so khong co trong dau vao: {sorted(extra)}")

    summary = prose.get("summary", "")
    if BARE_VERDICT.search(summary):                                     # C4
        fails.append(f"C4 phan quyet tran trui trong summary: {summary[:80]!r}")
    if ENROL_ADVICE.search(summary + " " + " ".join(c.get("text", "") for c in claims)):
        fails.append("C4 co loi khuyen ghi danh")
    if extra_sum := (numbers(summary) - src_nums):
        fails.append(f"C3 summary co so la: {sorted(extra_sum)}")
    return fails


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else SAMPLES
    blob = json.load(open(src, encoding="utf-8"))
    samples = blob["samples"]
    print(f"Nguon: {src}")
    results, n_claims = [], 0
    for s in samples:
        f = audit_one(s)
        n_claims += len(s["prose"].get("claims", []))
        results.append({"topic": s["topic"], "nct": s["nct"],
                        "n_claims": len(s["prose"].get("claims", [])),
                        "fails": f})

    bad = [r for r in results if r["fails"]]
    print(f"{len(samples)} output, {n_claims} claim\n")
    for r in bad:
        print(f"  {r['topic']}/{r['nct']}")
        for f in r["fails"]:
            print(f"      {f}")
    passed = len(samples) - len(bad)
    print(f"\n{passed}/{len(samples)} output sach"
          f"{'' if bad else '  — DAT exit criterion specs/09'}")

    json.dump({"n_samples": len(samples), "n_claims": n_claims,
               "n_clean": passed, "results": results},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Da ghi {OUT}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
