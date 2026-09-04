"""Statistical significance via paired bootstrap.

With 75 topics (dev) or 50 (test), nDCG differences under ~0.02 are easily
noise — and the ladder from Phase 3..8 produces several differences right in
that range. Without a test, "this rung beats that rung" is indistinguishable
from "two different runs".
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Compare two systems on the same set of topics.

    `a` and `b` are {topic_id: score}. Only topics present in BOTH are used —
    comparing on mismatched topic sets is meaningless.

    Returns the observed difference, a 95% CI (bootstrap percentile), and a
    two-sided p-value via the null-shift method: shift the difference
    distribution to 0, then count how many resamples have |diff| >= the
    observed one.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0, "diff": 0.0, "p": 1.0, "ci_lo": 0.0, "ci_hi": 0.0}

    d = np.array([a[t] - b[t] for t in shared], dtype=float)
    obs = float(d.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = d[idx].mean(axis=1)

    centred = draws - obs                      # null hypothesis: difference = 0
    p = float((np.abs(centred) >= abs(obs)).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])

    return {"n": len(shared), "diff": obs, "p": p,
            "ci_lo": float(lo), "ci_hi": float(hi)}


def compare(per_a: dict[str, dict[str, float]], per_b: dict[str, dict[str, float]],
            measures: list[str], name_a: str = "A", name_b: str = "B") -> str:
    lines = [f"\n{name_a}  vs  {name_b}", "=" * 72,
             f"{'do':34s} {'A':>8s} {'B':>8s} {'chenh':>8s} {'p':>7s}  KL"]
    for m in measures:
        if m not in per_a or m not in per_b:
            continue
        r = paired_bootstrap(per_a[m], per_b[m])
        ma = sum(per_a[m].values()) / max(len(per_a[m]), 1)
        mb = sum(per_b[m].values()) / max(len(per_b[m]), 1)
        verdict = "co y nghia" if r["p"] < 0.05 else "khong phan biet duoc"
        lines.append(f"{m:34s} {ma:8.4f} {mb:8.4f} {r['diff']:+8.4f} "
                     f"{r['p']:7.4f}  {verdict}")
    return "\n".join(lines)
