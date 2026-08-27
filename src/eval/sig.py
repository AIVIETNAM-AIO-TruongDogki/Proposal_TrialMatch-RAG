"""Kiem dinh y nghia thong ke bang paired bootstrap.

Voi 75 topic (dev) hoac 50 topic (test), chenh lech nDCG duoi khoang 0.02 rat
de la nhieu — ma cai thang do cua Phase 3..8 se sinh ra vai chenh lech dung
trong khoang do. Khong co kiem dinh thi khong phan biet duoc "rung nay cao hon
rung kia" voi "hai lan chay khac nhau".
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """So sanh hai he thong tren cung tap topic.

    `a` va `b` la {topic_id: diem}. Chi dung cac topic co mat o CA HAI ben —
    so sanh tren tap topic lech nhau la vo nghia.

    Tra ve chenh lech quan sat, khoang tin cay 95% (bootstrap percentile), va
    p-value hai phia theo phuong phap dich ve gia thuyet khong (shift method):
    dich phan phoi chenh lech ve 0 roi dem xem bao nhieu lan resample cho gia
    tri tuyet doi >= chenh lech quan sat.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0, "diff": 0.0, "p": 1.0, "ci_lo": 0.0, "ci_hi": 0.0}

    d = np.array([a[t] - b[t] for t in shared], dtype=float)
    obs = float(d.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = d[idx].mean(axis=1)

    centred = draws - obs                      # gia thuyet khong: chenh lech = 0
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
