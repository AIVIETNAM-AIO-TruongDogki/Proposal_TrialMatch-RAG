"""Phase 8 buoc 3 — bien invariant 3 tu loi hua thanh phep do.

MOT QUYET DINH KHONG TRICH DAN DUOC LA MOT QUYET DINH BI VUT
-------------------------------------------------------------
`criterion_quote` phai la chuoi con nguyen van cua CHINH tieu chi do — kiem
qua store.verify_quote(), ham nay doi chieu nguoc ve span_start/span_end trong
trials.criteria_raw, tuc la ve dung chuoi goc lay tu XML ClinicalTrials.gov.
Khong phai doi chieu voi mot ban da bi chinh sua o dau do.

`patient_evidence` phai la chuoi con nguyen van cua BENH AN.

KIEM CA HAI PHIA — KIEM MOT PHIA LA LOT NUA SO LOI
---------------------------------------------------
Model co the trich dung tieu chi roi bia bang chung benh nhan (de ket luan
`satisfied` cho mot dieu benh an khong he noi), hoac trich dung benh an roi
gan cho mot tieu chi khac. Hai loi doc lap nhau, nen hai phep kiem doc lap.

TY LE BI VUT LA MOT KET QUA PHAI BAO CAO
-----------------------------------------
Khong phai mot bo loc am tham. "Ty le vi pham grounding = 6%" la mot con so
do duoc ve do trung thuc cua model, va no thuoc ve bang ket qua cuoi cung.

NGOAI LE CO Y: `unverifiable` duoc phep co patient_evidence RONG
-----------------------------------------------------------------
Khong the trich dan bang chung cho mot thu benh an khong nhac toi. Bat buoc
trich dan o day se ep model bia ra mot doan van — dung dieu ta dang chong.
"""

from __future__ import annotations

from src.corpus import store
from src.reasoning.schema import LABELS


def norm(s: str) -> str:
    """Chuan hoa y het store.verify_quote(): gop khoang trang, ha chu thuong."""
    return " ".join((s or "").split()).lower()


def grounded_in(quote: str, source: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(source)


def check(conn, decision: dict, nct_id: str, idx: int, narrative: str
          ) -> tuple[bool, str | None]:
    """(hop le, ly do bi vut). Ly do duoc giu lai de phan tich, khong chi dem."""
    label = decision.get("label")
    if label not in LABELS:
        return False, "label_invalid"

    cq = decision.get("criterion_quote") or ""
    if not store.verify_quote(conn, nct_id, idx, cq):
        return False, "criterion_quote_not_in_criterion"

    pe = decision.get("patient_evidence") or ""
    if label == "unverifiable":
        # Co y cho phep rong: khong the trich dan cho thu khong duoc nhac toi.
        if pe and not grounded_in(pe, narrative):
            return False, "patient_evidence_not_in_narrative"
        return True, None

    if not pe.strip():
        return False, "patient_evidence_empty"
    if not grounded_in(pe, narrative):
        return False, "patient_evidence_not_in_narrative"
    return True, None


def summarize(rejections: dict[str, int], total: int) -> str:
    if not total:
        return "  (khong co quyet dinh nao)"
    bad = sum(rejections.values())
    lines = [f"  {total:,} quyet dinh, {bad:,} bi vut vi khong trich dan duoc "
             f"({bad/total*100:.1f}%)"]
    for reason, n in sorted(rejections.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {reason:38s} {n:6,}  ({n/total*100:.1f}%)")
    return "\n".join(lines)
