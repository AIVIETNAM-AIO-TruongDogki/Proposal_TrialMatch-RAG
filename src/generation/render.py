"""Phase 9 (khung cau truc) / Phase 10 (dung that) — render KHONG goi LLM.

`specs/09-evidence-grounded-generation.md` dac ta mot buoc danh bong van xuoi
bang LLM. Ban demo song nay CO TINH BO qua LLM-polish do: moi request da tra
1 (trich xuat) + toi da LIVE_TOP_N (suy luan) loi goi Gemini, them mot loi
goi/trial nua cho van phong se nhan doi chi phi quota cho mot tinh nang da
cang quota — xem "Pham vi" trong ke hoach.

Nhung con lai cua specs/09 la CAU TRUC, khong phai van phong, va no ap dung
duoc bang mau thuan tuy:
  - buoc 3: `unverifiable` phai hien ro thanh mot dong giai thich, khong phai
    badge trong rong (invariant 1)
  - buoc 4: moi claim di kem trich dan tieu chi cua no (invariant 3)
  - buoc 5: khung ngon ngu la "ho tro quyet dinh", khong bao gio "eligible"
    (invariant 4)

Ham o day chi sap xep lai du lieu CO SAN tu Phase 8 (`aggregate`/`reason`) va
Phase 1 (`store`) thanh mot dict de JSON-hoa gui qua SSE — khong doan them,
khong tao chu nao khong co trong du lieu goc.
"""

from __future__ import annotations

UNVERIFIABLE_NOTE = "cannot be determined from the available information"

LABEL_COPY = {
    "satisfied": "satisfied",
    "violated": "violated",
    "unverifiable": f"unverifiable — {UNVERIFIABLE_NOTE}",
}

DISCLAIMER = (
    "Decision support only — not a clinical eligibility determination. "
    "Review with a clinician or trial coordinator before acting on this output."
)


def criterion_row(d: dict) -> dict:
    """Mot dong bang tieu chi, dung dinh dang da qua `verify.check()` phia server."""
    return {
        "criterion_idx": d["criterion_idx"],
        "section": d.get("section", "unknown"),
        "label": d["label"],
        "label_display": LABEL_COPY.get(d["label"], d["label"]),
        "criterion_quote": d.get("criterion_quote", ""),
        "patient_evidence": d.get("patient_evidence", ""),
        "reasoning": d.get("reasoning", ""),
    }


def trial_card(trial: dict, decisions: list[dict], score: float) -> dict:
    """Mot the trial da xep hang, san sang JSON-hoa gui qua SSE.

    `trial` la dict tu `store.get_trial()`. `decisions` la ket qua da qua kiem
    chung cua `reason.run_batch_trial()`. Khong goi Gemini o day.
    """
    rows = [criterion_row(d) for d in sorted(decisions, key=lambda x: x["criterion_idx"])]
    n_vio = sum(1 for r in rows if r["label"] == "violated")
    n_unv = sum(1 for r in rows if r["label"] == "unverifiable")
    n_sat = sum(1 for r in rows if r["label"] == "satisfied")
    return {
        "nct_id": trial["nct_id"],
        "title": trial.get("title") or "",
        "phase": trial.get("phase") or "",
        "status": trial.get("status") or "",
        "score": score,
        "n_criteria": len(rows),
        "n_satisfied": n_sat,
        "n_violated": n_vio,
        "n_unverifiable": n_unv,
        "criteria": rows,
    }
