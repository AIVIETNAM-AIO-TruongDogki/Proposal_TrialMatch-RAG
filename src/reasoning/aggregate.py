"""Phase 8 buoc 5 — gop quyet dinh muc tieu chi thanh quyet dinh muc TRIAL.

LUAT GOP LA THAM SO TU DO, KHONG PHAI CHI TIET CAI DAT
-------------------------------------------------------
No AN HUONG TRUC TIEP toi con so tieu de. Neu chon mot luat roi giau trong
code thi bao cao cuoi se khong phan biet duoc "suy luan eligibility hoat dong"
voi "luat gop nay tinh co hop voi qrels". Vi vay: luat duoc dat ten, khai bao
tuong minh, va CO THE THAY BANG CO — de Phase 11 ablate chinh no.

`unverifiable` KHONG duoc coi la vi pham
-----------------------------------------
Do la ca diem cua ba trang thai. Mot trial co 5 tieu chi khong kiem chung duoc
thi KHONG bi loai — no bi xep sau mot trial da kiem du, nhung van con trong
danh sach. Coi `unverifiable` nhu `violated` la quay ve nhi phan bang cua sau.
"""

from __future__ import annotations

import re

RULES = ("strict", "lenient", "count")

DISQUALIFIED = 0.0    # bi loai — gia tri RIENG, khong trial nao khac cham toi
_MIN = 1e-6           # san cho trial KHONG bi loai

# Tieu chi mo dau bang mot phu dinh la tieu chi LOAI TRU ve mat chuc nang, du
# blob goc khong co header "Exclusion Criteria:". Dinh dang NCI cu viet ca hai
# loai thanh mot danh sach phang duoi cac header nhu "PATIENT CHARACTERISTICS:",
# phan biet nhau bang the phu dinh: "Histologically confirmed X" la thu nhan,
# "No prior chemotherapy" la loai tru. Parser gan `unknown` cho ca hai — dung
# theo nghia CAU TRUC (khong co header), nhung `lenient` doc no theo nghia
# NGU NGHIA va vi vay bo sot nhung tieu chi loai tru that.
_NEG_LEAD_RE = re.compile(r"^\s*(?:no|not|none|without|absence of|free of)\b", re.I)


def effective_section(d: dict) -> str | None:
    """Section dung cho luat gop — suy ra cuc phu dinh khi parser tra `unknown`.

    Tach roi khoi `section` da luu trong DB CO CHU DICH: `section` la su that
    ve CAU TRUC (criterion nay nam duoi header nao), con ham nay la mot suy
    dien NGU NGHIA. Tron hai thu vao nhau trong DB se lam mat kha nang do xem
    suy dien nay dung bao nhieu. Vi vay no song o day, co ten, va Phase 11
    ablate duoc bang cach goi `trial_score(..., infer_section=False)`.

    Doc `criterion_quote` chu khong doc van ban tieu chi goc, de `trial_score`
    khong can ket noi DB. Do tren smoke test: 98,6% trich dan bat dau dung tu
    dau tieu chi, nen proxy nay sai ~1,4% — va chi ap dung cho 4,5% tieu chi
    mang nhan `unknown`, tuc phoi nhiem tong ~0,06%.
    """
    sec = d.get("section")
    if sec in ("inclusion", "exclusion"):
        return sec
    if _NEG_LEAD_RE.match(d.get("criterion_quote") or ""):
        return "exclusion"
    return sec


def _spread(raw: float, lo: float, hi: float) -> float:
    """Anh xa `raw` tuyen tinh vao [_MIN, 1] — don dieu, khong bao gio cham 0.

    Ban truoc tra `max(base - penalty, 1e-6)`, va `base - penalty` AM bat cu
    khi nao `unv > 2*sat`. Voi `unverifiable` chiem ~68% nhan — dung nhu thiet
    ke ba trang thai mong doi — dieu do la BINH THUONG chu khong phai ngoai le:
    ca ba trial trong smoke test deu cham san. Moi trial hoa nhau o 1e-6 thi
    xep hang roi het ve tie-break truy hoi va tang eligibility do duoc BANG 0.
    Ep vao [_MIN, 1] giu nguyen thu tu ma van tach duoc cac trial ra.
    """
    z = (raw - lo) / (hi - lo)
    return _MIN + (1.0 - _MIN) * min(max(z, 0.0), 1.0)


def trial_score(decisions: list[dict], rule: str = "strict",
                infer_section: bool = True) -> float:
    """Diem xep hang cua mot trial tu cac quyet dinh tieu chi da qua kiem chung.

    strict  — mot `violated` bat ky (o BAT KY muc nao) => loai (diem 0).
              Con lai: uu tien trial co nhieu tieu chi da xac nhan `satisfied`
              va it `unverifiable`.
    lenient — chi `violated` o tieu chi LOAI TRU moi loai. Vi pham tieu chi
              nhan (inclusion) chi bi tru diem. Phan anh thuc te lam sang:
              tieu chi nhan thuong mem hon tieu chi loai tru.
    count   — khong loai ai ca; xep hang THUAN theo ty le satisfied, khong
              phat `unverifiable`. Duong co so de biet phan "loai bo" dong gop
              bao nhieu; neu no cong them hinh phat thi no khong con la duong
              co so nua ma la mot luat gop thu ba.

    `infer_section=False` tat suy dien cuc phu dinh — de ablate xem no dang
    doi bao nhieu ket luan cua `lenient`.
    """
    if not decisions:
        return DISQUALIFIED
    n = len(decisions)
    sat = sum(1 for d in decisions if d["label"] == "satisfied")
    unv = sum(1 for d in decisions if d["label"] == "unverifiable")
    vio_any = sum(1 for d in decisions if d["label"] == "violated")
    sec_of = effective_section if infer_section else (lambda d: d.get("section"))
    vio_exc = sum(1 for d in decisions
                  if d["label"] == "violated" and sec_of(d) == "exclusion")

    if rule == "strict" and vio_any:
        return DISQUALIFIED
    if rule == "lenient" and vio_exc:
        return DISQUALIFIED

    if rule == "count":
        return _spread(sat / n, 0.0, 1.0)

    # Trial kiem duoc nhieu hon duoc xep truoc trial phan lon la unverifiable.
    # `unverifiable` chi HA THAP thu hang, khong loai — xem docstring module.
    raw = (sat - 0.5 * unv) / n
    lo = -0.5
    if rule == "lenient":
        raw -= 0.3 * ((vio_any - vio_exc) / n)
        lo = -0.8
    return _spread(raw, lo, 1.0)


def rerank_by_eligibility(base_run: dict, decisions_by: dict, rule: str = "strict",
                          keep_unjudged: bool = True) -> dict:
    """Xep lai run bang diem eligibility, giu diem truy hoi lam tie-break.

    Trial khong co quyet dinh nao (ngoai top-N da suy luan) duoc giu NGUYEN
    thu tu phia sau thay vi bi vut: vut chung se lam recall tut mot cach gia
    tao va lam bac 5 trong tot hon thuc te.
    """
    out: dict[str, dict[str, float]] = {}
    for tid, docs in base_run.items():
        ranked = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))
        n = len(ranked)
        new: dict[str, float] = {}
        for i, (nct, ret_score) in enumerate(ranked):
            key = (tid, nct)
            if key in decisions_by:
                elig = trial_score(decisions_by[key], rule)
                # Trial da suy luan luon dung TREN trial chua suy luan cung diem,
                # va thu tu truy hoi goc pha the hoa trong noi bo nhom.
                new[nct] = 1000.0 + elig * 100.0 + (n - i) / (n + 1)
            elif keep_unjudged:
                new[nct] = (n - i) / (n + 1)
        out[tid] = new
    return out
