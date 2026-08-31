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

RULES = ("strict", "lenient", "count")


def trial_score(decisions: list[dict], rule: str = "strict") -> float:
    """Diem xep hang cua mot trial tu cac quyet dinh tieu chi da qua kiem chung.

    strict  — mot `violated` bat ky (o BAT KY muc nao) => loai (diem 0).
              Con lai: uu tien trial co nhieu tieu chi da xac nhan `satisfied`
              va it `unverifiable`.
    lenient — chi `violated` o tieu chi LOAI TRU moi loai. Vi pham tieu chi
              nhan (inclusion) chi bi tru diem. Phan anh thuc te lam sang:
              tieu chi nhan thuong mem hon tieu chi loai tru.
    count   — khong loai ai ca; xep hang thuan theo ty le satisfied.
              Duong co so de biet phan "loai bo" dong gop bao nhieu.
    """
    if not decisions:
        return 0.0
    n = len(decisions)
    sat = sum(1 for d in decisions if d["label"] == "satisfied")
    unv = sum(1 for d in decisions if d["label"] == "unverifiable")
    vio_exc = sum(1 for d in decisions
                  if d["label"] == "violated" and d.get("section") == "exclusion")
    vio_any = sum(1 for d in decisions if d["label"] == "violated")

    if rule == "strict" and vio_any:
        return 0.0
    if rule == "lenient" and vio_exc:
        return 0.0

    # Trial kiem duoc nhieu hon duoc xep truoc trial phan lon la unverifiable.
    # `unverifiable` chi HA THAP thu hang, khong loai — xem docstring module.
    base = sat / n
    penalty = 0.5 * (unv / n)
    if rule == "lenient":
        penalty += 0.3 * ((vio_any - vio_exc) / n)
    return max(base - penalty, 1e-6)


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
