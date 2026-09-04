"""Phase 11 follow-up — what does a relevance check on grounding actually cost?

    PYTHONPATH=. .venv/bin/python scripts/phase11_relevance_check.py

`verify.grounded_in` only asks whether a quote is verbatim. `verify.supports`
additionally asks whether it bears on the criterion. This applies the stricter
rule to the decisions ALREADY cached for dev 2021 — it can only drop decisions,
never add them, so no API call is needed — then re-scores rung 5 both ways.

Two variants are measured, because the carve-out is the whole design question:
without it a stated sex or age can no longer clear a criterion the patient
cannot meet by definition. Reports Macro-F1 (the project's primary metric)
alongside contamination, since a rule that cuts contamination by disqualifying
more trials is the exact failure the three-state design exists to avoid.
"""

from __future__ import annotations

import json

from src.eval import data, metrics, run_io, sig
from src.reasoning import aggregate, score as rscore, verify

CACHE = "data/reasoning/2021.gemini-3.5-flash-lite.trial.json"
BASE_RUN = "runs/hybrid.dev.txt"
RULE = "strict"


def filtered(dec: dict, exempt: bool) -> tuple[dict, int, int]:
    """Drop decisions whose evidence does not bear on the criterion."""
    out, kept, dropped = {}, 0, 0
    for key, ds in dec.items():
        keep = []
        for d in ds:
            if d["label"] == "unverifiable":     # empty evidence is legitimate here
                keep.append(d); kept += 1; continue
            if verify.supports(d.get("patient_evidence", ""),
                               d.get("criterion_quote", ""), exempt_demographic=exempt):
                keep.append(d); kept += 1
            else:
                dropped += 1
        out[key] = keep
    return out, kept, dropped


def main() -> int:
    dec, _ = rscore.load_decisions(CACHE)
    qrels = data.load_qrels(2021)
    base = run_io.read_run(BASE_RUN)

    variants = [("hien tai (chi nguyen van)", dec, 0)]
    for exempt in (True, False):
        f, _, dropped = filtered(dec, exempt)
        variants.append((f"+ lien quan ({'co' if exempt else 'khong'} mien tru nhan khau)",
                         f, dropped))

    print(f"{'bien the':38s} {'vut them':>9s} {'F1':>7s} {'acc':>7s} "
          f"{'contam@10':>10s} {'eligN@10':>9s} {'rec@1000':>9s}")
    print("-" * 94)
    runs = {}
    for name, d, dropped in variants:
        e = rscore.trial_level_eval(d, qrels, RULE)
        r = aggregate.rerank_by_eligibility(base, d, RULE)
        a = metrics.aggregate(metrics.evaluate(r, qrels))
        runs[name] = (r, a)
        print(f"{name:38s} {dropped:9,} {e['f1']:7.4f} {e['accuracy']:7.4f} "
              f"{a['elig/contamination_10']:10.4f} {a['eligible/ndcg_cut_10']:9.4f} "
              f"{a['elig/recall_1000']:9.4f}")

    base_name = variants[0][0]
    per_base = metrics.evaluate(runs[base_name][0], qrels)
    for name, _, _ in variants[1:]:
        per = metrics.evaluate(runs[name][0], qrels)
        print(sig.compare(per, per_base,
                          ["eligible/ndcg_cut_10", "eligible/P_10",
                           "elig/contamination_10", "elig/recall_1000"],
                          name_a=name, name_b=base_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
