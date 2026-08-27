"""Hai ho do, luon bao cao canh nhau.

HO 1 — CHINH THUC (de so sanh voi cac bai bao da cong bo)
    Dung qrels nguyen ban. Da xac minh bang thuc nghiem tren pytrec_eval:
      * ndcg dung gain TUYEN TINH (gain = rel), khong phai 2^rel - 1;
      * P / recall / recip_rank la NHI PHAN voi nguong rel > 0.
    He qua: mot trial EXCLUDED(1) duoc tinh la HIT trong P@10 va MRR, va duoc
    gain duong trong nDCG.

HO 2 — NHAN THUC ELIGIBILITY (de do luan diem cua de tai)
    Dung qrels da ban do lai: chi ELIGIBLE moi co gain. Cong them
    `contamination@k` — ty le top-k la trial lien quan y khoa nhung bi loai tru.
    Day moi la con so ma Phase 8 phai lam giam.

Mot he thong loc eligibility tot se lam ho 2 tot len va co the lam ho 1 XAU di.
Do la ket qua dung, khong phai loi. Nham lan hai dieu nay se dan den viec vut bo
chinh dong gop cua de tai.
"""

from __future__ import annotations

import pytrec_eval

from src.eval.data import ELIGIBLE, EXCLUDED, Qrels, eligible_only

Run = dict[str, dict[str, float]]

MEASURES = {"ndcg_cut.10", "ndcg_cut.100", "P.10", "recall.1000", "recip_rank",
            # bpref chi dem thu tu giua cac tai lieu DA duoc cham, nen no on dinh
            # khi judgment khong day du. Xem `condense()` ben duoi.
            "bpref"}


def _rank(run_topic: dict[str, float]) -> list[str]:
    """Xep hang theo diem giam dan; hoa thi pha bang doc-id de tai lap duoc."""
    return [d for d, _ in sorted(run_topic.items(), key=lambda kv: (-kv[1], kv[0]))]


def contamination_at_k(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    """Ty le trong top-k la trial EXCLUDED — cang thap cang tot.

    Mau so la k, khong phai so tai lieu da duoc cham. Tai lieu chua cham duoc
    coi la khong gay o nhiem. Vi vay LUON doc kem `judged_at_k`: mot he thong
    tra ve toan tai lieu ngoai pool se co contamination thap mot cach gia tao.
    """
    out = {}
    for tid in qrels:
        top = _rank(run.get(tid, {}))[:k]
        rel = qrels[tid]
        out[tid] = sum(1 for d in top if rel.get(d) == EXCLUDED) / k if k else 0.0
    return out


def judged_at_k(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    """Ty le top-k nam trong pool da cham. Do phu, khong phai chat luong."""
    out = {}
    for tid in qrels:
        top = _rank(run.get(tid, {}))[:k]
        out[tid] = sum(1 for d in top if d in qrels[tid]) / k if k else 0.0
    return out


def eligible_recall(run: Run, qrels: Qrels, k: int = 1000) -> dict[str, float]:
    """Recall chi tinh tren trial ELIGIBLE — tran cua moi tang xep hang phia sau."""
    out = {}
    for tid in qrels:
        gold = {d for d, r in qrels[tid].items() if r == ELIGIBLE}
        if not gold:
            continue
        top = set(_rank(run.get(tid, {}))[:k])
        out[tid] = len(top & gold) / len(gold)
    return out


def condense(run: Run, qrels: Qrels) -> Run:
    """Bo moi tai lieu CHUA duoc cham ra khoi run truoc khi cham diem.

    Ly do: qrels cua TREC duoc tao bang pooling — chi ~708/375.580 thu nghiem
    moi benh nhan tung duoc bac si xem. Tai lieu ngoai pool mac dinh bi coi la
    KHONG lien quan. Nen mot he thong tim ra thu nghiem that su phu hop nhung
    khong doi nao nam 2022 tim ra se bi PHAT OAN.

    "Condensed list" (Sakai 2007) xu ly bang cach xoa han cac tai lieu chua cham
    khoi bang xep hang, roi cham tren phan con lai. Cau hoi doi thanh: "trong so
    nhung thu da duoc cham, ban xep dung thu tu den dau?" — cau hoi nay khong bi
    thien lech boi do sau cua pool.

    Doc kem `judged@k`: neu judged@10 cao thi hai cach cham gan nhu trung nhau
    va thien lech khong dang ke; neu thap thi diem chinh thuc dang bi danh gia
    thap mot cach he thong, va bao cao cuoi phai noi ro dieu do.
    """
    return {t: {d: sc for d, sc in docs.items() if d in qrels.get(t, {})}
            for t, docs in run.items()}


def _pytrec(run: Run, qrels: Qrels) -> dict[str, dict[str, float]]:
    # pytrec_eval bo qua topic khong co tai lieu lien quan nao; giu nguyen hanh vi do.
    ev = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    return ev.evaluate(run)


def evaluate(run: Run, qrels: Qrels) -> dict[str, dict[str, float]]:
    """Tra ve {ten_do: {topic_id: diem}} cho CA HAI ho do."""
    per: dict[str, dict[str, float]] = {}

    for label, q in (("official", qrels), ("eligible", eligible_only(qrels))):
        res = _pytrec(run, q)
        for tid, scores in res.items():
            for m, v in scores.items():
                per.setdefault(f"{label}/{m}", {})[tid] = v

    # Cham lai tren danh sach da bo tai lieu chua duoc cham (chong pool bias).
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


# Thu tu in ra. Hai ho tach roi de khong ai vo tinh doc nham dong.
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
