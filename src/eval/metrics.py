"""Two metric families, always reported side by side.

FAMILY 1 — OFFICIAL (for comparing against published papers)
    Uses qrels as-is. Verified empirically against pytrec_eval:
      * ndcg uses LINEAR gain (gain = rel), not 2^rel - 1;
      * P / recall / recip_rank are BINARY with threshold rel > 0.
    Consequence: an EXCLUDED(1) trial counts as a HIT in P@10 and MRR, and
    gets positive gain in nDCG.

FAMILY 2 — ELIGIBILITY-AWARE (measures this project's actual claim)
    Uses remapped qrels: only ELIGIBLE gets gain. Adds `contamination@k` —
    the fraction of top-k that's medically relevant but excluded. This is
    the number Phase 8 must bring down.

A good eligibility filter improves family 2 and can make family 1 WORSE. That
is the correct result, not a bug. Confusing the two would mean discarding
this project's actual contribution.
"""

from __future__ import annotations

import pytrec_eval

from src.eval.data import ELIGIBLE, EXCLUDED, Qrels, eligible_only

Run = dict[str, dict[str, float]]

MEASURES = {"ndcg_cut.10", "ndcg_cut.100", "P.10", "recall.1000", "recip_rank",
            # bpref only counts order among JUDGED documents, so it stays
            # stable under incomplete judgments. See `condense()` below.
            "bpref"}


def _rank(run_topic: dict[str, float]) -> list[str]:
    """Rank by descending score; ties broken by doc-id for reproducibility."""
    return [d for d, _ in sorted(run_topic.items(), key=lambda kv: (-kv[1], kv[0]))]


def contamination_at_k(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    """Fraction of top-k that's an EXCLUDED trial — lower is better.

    Denominator is k, not the number of judged documents. Unjudged documents
    count as non-contaminating. ALWAYS read alongside `judged_at_k`: a system
    returning only out-of-pool documents would show artificially low contamination.
    """
    out = {}
    for tid in qrels:
        top = _rank(run.get(tid, {}))[:k]
        rel = qrels[tid]
        out[tid] = sum(1 for d in top if rel.get(d) == EXCLUDED) / k if k else 0.0
    return out


def judged_at_k(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    """Fraction of top-k inside the judged pool. Measures coverage, not quality."""
    out = {}
    for tid in qrels:
        top = _rank(run.get(tid, {}))[:k]
        out[tid] = sum(1 for d in top if d in qrels[tid]) / k if k else 0.0
    return out


def eligible_recall(run: Run, qrels: Qrels, k: int = 1000) -> dict[str, float]:
    """Recall computed only over ELIGIBLE trials — the ceiling for every later ranking stage."""
    out = {}
    for tid in qrels:
        gold = {d for d, r in qrels[tid].items() if r == ELIGIBLE}
        if not gold:
            continue
        top = set(_rank(run.get(tid, {}))[:k])
        out[tid] = len(top & gold) / len(gold)
    return out


def condense(run: Run, qrels: Qrels) -> Run:
    """Remove every UNJUDGED document from the run before scoring.

    TREC qrels are built by pooling — only ~708/375,580 trials per patient
    were ever reviewed by a physician, and anything outside the pool defaults
    to non-relevant. A system finding a genuinely good match no 2022 pooled
    system ever surfaced gets unfairly PENALIZED.

    The "condensed list" method (Sakai 2007) strips unjudged documents from
    the ranking before scoring, turning the question into "among judged
    documents, how well is the order?" — a question the pool's depth can't bias.

    Read alongside `judged@k`: high judged@10 means the two scoring methods
    nearly agree; low judged@10 means the official score is systematically
    underrating the system, and the final report must say so.
    """
    return {t: {d: sc for d, sc in docs.items() if d in qrels.get(t, {})}
            for t, docs in run.items()}


def _pytrec(run: Run, qrels: Qrels) -> dict[str, dict[str, float]]:
    # pytrec_eval skips topics with no relevant documents; that behavior is kept as-is.
    ev = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    return ev.evaluate(run)


def evaluate(run: Run, qrels: Qrels) -> dict[str, dict[str, float]]:
    """Returns {metric_name: {topic_id: score}} for BOTH metric families."""
    per: dict[str, dict[str, float]] = {}

    for label, q in (("official", qrels), ("eligible", eligible_only(qrels))):
        res = _pytrec(run, q)
        for tid, scores in res.items():
            for m, v in scores.items():
                per.setdefault(f"{label}/{m}", {})[tid] = v

    # Rescore on the pool-bias-free (unjudged documents removed) list.
    cond = condense(run, eligible_only(qrels))
    for tid, scores in _pytrec(cond, eligible_only(qrels)).items():
        for m, v in scores.items():
            per.setdefault(f"cond/{m}", {})[tid] = v

    per["elig/contamination_10"] = contamination_at_k(run, qrels, 10)
    per["elig/contamination_100"] = contamination_at_k(run, qrels, 100)
    per["elig/judged_10"] = judged_at_k(run, qrels, 10)
    per["elig/recall_1000"] = eligible_recall(run, qrels, 1000)
    return per


def aggregate(per_topic: dict[str, dict[str, float]]) -> dict[str, float]:
    return {m: (sum(v.values()) / len(v) if v else 0.0) for m, v in per_topic.items()}


# Print order. The two families are kept visually separate so no one misreads a row.
REPORT_ORDER = [
    ("CHINH THUC (excluded=1 duoc tinh diem)", [
        ("official/ndcg_cut_10",   "nDCG@10"),
        ("official/ndcg_cut_100",  "nDCG@100"),
        ("official/P_10",          "P@10"),
        ("official/recip_rank",    "MRR"),
        ("official/recall_1000",   "Recall@1000"),
    ]),
    ("NHAN THUC ELIGIBILITY (chi eligible duoc tinh diem)", [
        ("eligible/ndcg_cut_10",   "nDCG@10 (eligible-only)"),
        ("eligible/P_10",          "P@10  (eligible-only)"),
        ("eligible/recip_rank",    "MRR   (eligible-only)"),
        ("elig/recall_1000",       "Recall@1000 (eligible-only)"),
        ("elig/contamination_10",  "Contamination@10  [THAP = TOT]"),
        ("elig/contamination_100", "Contamination@100 [THAP = TOT]"),
        ("elig/judged_10",         "Judged@10 (do phu pool)"),
    ]),
    ("CHONG POOL BIAS (chi xet tai lieu DA duoc cham)", [
        ("cond/ndcg_cut_10",       "nDCG@10 condensed (eligible-only)"),
        ("cond/P_10",              "P@10  condensed (eligible-only)"),
        ("eligible/bpref",         "bpref (eligible-only)"),
    ]),
]


def format_report(agg: dict[str, float], n_topics: int, title: str = "") -> str:
    lines = []
    if title:
        lines += [f"\n{title}", "=" * 64]
    lines.append(f"{n_topics} topic")
    for header, rows in REPORT_ORDER:
        lines.append(f"\n  {header}")
        for key, label in rows:
            if key in agg:
                lines.append(f"    {label:34s} {agg[key]:.4f}")
    return "\n".join(lines)
