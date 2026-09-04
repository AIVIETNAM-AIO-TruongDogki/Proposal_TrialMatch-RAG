"""Phase 11 — paired-bootstrap significance across the full ablation ladder.

    PYTHONPATH=. .venv/bin/python scripts/phase11_significance.py

Reads only the per-topic scores already stored in results/rung*.json, so it
re-scores nothing and touches no run file — the 2022 test set stays scored
exactly once. Writes results/_sig_ladder.2022.json.
"""

from __future__ import annotations

import json

from src.eval import sig

MEASURES = ["official/ndcg_cut_10", "eligible/ndcg_cut_10", "eligible/P_10",
            "elig/contamination_10", "elig/recall_1000"]

RUNGS = {
    "rung1_lexical": "results/rung1_lexical.json",
    "rung2_dense": "results/rung2_dense.json",
    "rung3_hybrid": "results/rung3_hybrid.json",
    "rung4_rerank": "results/rung4_rerank.json",
    "rung5_eligibility": "results/rung5_eligibility.json",
}

# (a, b) reads "does a beat b". The headline comparison is last so it lands at
# the bottom of the printed report.
PAIRS = [
    ("rung2_dense", "rung1_lexical"),
    ("rung3_hybrid", "rung1_lexical"),
    ("rung3_hybrid", "rung2_dense"),
    ("rung4_rerank", "rung3_hybrid"),
    ("rung5_eligibility", "rung4_rerank"),
    ("rung5_eligibility", "rung3_hybrid"),
]

OUT = "results/_sig_ladder.2022.json"


def main() -> int:
    loaded = {k: json.load(open(v, encoding="utf-8")) for k, v in RUNGS.items()}
    out = {}
    for a, b in PAIRS:
        pa, pb = loaded[a]["per_topic"], loaded[b]["per_topic"]
        print(sig.compare(pa, pb, MEASURES, name_a=a, name_b=b))
        out[f"{a}__vs__{b}"] = {m: sig.paired_bootstrap(pa[m], pb[m])
                                for m in MEASURES if m in pa and m in pb}

    json.dump({"year": 2022, "n_boot": 10000, "seed": 0, "comparisons": out},
              open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\n\nDa ghi: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
