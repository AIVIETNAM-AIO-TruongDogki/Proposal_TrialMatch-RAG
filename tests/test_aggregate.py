"""Ghim hai loi da lam sai KET QUA Phase 8 ma khong nem ngoai le nao.

    python tests/test_aggregate.py

Ca hai deu thuoc loai nguy hiem nhat: chung tra ve mot con so trong hop ly,
nen khong co gi bao dong. Chung chi lo ra khi doi chieu voi qrels tren dung ba
trial. Vi vay chung duoc ghim o day.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.reasoning.aggregate import (DISQUALIFIED, RULES, effective_section,
                                     trial_score)


def d(label, section="exclusion", quote="Prior chemotherapy"):
    return {"label": label, "section": section, "criterion_quote": quote}


def test_count_khong_sup_ve_san():
    """`count` la NHANH DOI CHUNG — no phai xep hang duoc, khong duoc hoa nhau.

    Ban cu: sat/n - 0.5*unv/n, am bat cu khi nao unv > 2*sat. Voi unverifiable
    chiem ~68% nhan (dung nhu thiet ke ba trang thai mong doi) do la truong hop
    THUONG, nen moi trial deu bi kep xuong 1e-6 va tang eligibility do duoc = 0.
    """
    it = [d("satisfied")] * 2 + [d("unverifiable")] * 8      # unv = 4x sat
                                                             # ban cu: -0.2 -> san
    nhieu_sat = [d("satisfied")] * 6 + [d("unverifiable")] * 4

    a, b = trial_score(it, "count"), trial_score(nhieu_sat, "count")
    assert a > DISQUALIFIED, f"count cham gia tri BI LOAI: {a}"
    assert b > a, f"count khong con xep hang duoc: {a} vs {b}"

    # va `count` phai la sat/n THUAN — khong phat unverifiable, neu khong no la
    # mot luat gop thu ba chu khong phai duong co so.
    assert abs(a - 0.2) < 1e-3, a
    assert abs(b - 0.6) < 1e-3, b


def test_strict_lenient_cung_khong_sup_ve_san():
    """Loi san khong chi o `count`, va trieu chung that KHONG phai la 'bang 0'
    ma la 'HOA NHAU'.

    Ban cu kep bang max(..., 1e-6), nen hai trial khac han nhau van ra CUNG
    mot so ngay khi ca hai co base < penalty. Do la luc xep hang chet: moi
    thu roi ve tie-break truy hoi va tang eligibility do duoc bang 0.
    """
    te = [d("satisfied")] + [d("unverifiable")] * 9          # cu: -0.35 -> san
    do_hon = [d("satisfied")] * 3 + [d("unverifiable")] * 7  # cu: -0.05 -> san
    for rule in ("strict", "lenient"):
        a, b = trial_score(te, rule), trial_score(do_hon, rule)
        assert a > DISQUALIFIED, f"{rule} lan vao gia tri BI LOAI: {a}"
        assert b > a, (f"{rule} hoa nhau giua hai trial khac han: {a} vs {b} "
                       f"— day chinh la loi san 1e-6")


def test_bi_loai_la_gia_tri_rieng():
    """Chi trial bi loai moi duoc mang 0.0 — neu khong thi khong phan biet
    duoc 'bi loai' voi 'diem thap', va ca luat gop mat y nghia."""
    assert trial_score([d("violated")], "strict") == DISQUALIFIED
    assert trial_score([], "strict") == DISQUALIFIED
    xau_nhat = [d("unverifiable")] * 10
    for rule in RULES:
        assert trial_score(xau_nhat, rule) > DISQUALIFIED


def test_suy_dien_cuc_phu_dinh():
    """Dinh dang NCI cu khong co header, phan biet bang the phu dinh.

    NCT00004259 that: 3 tieu chi loai tru ('No prior chemotherapy'...) mang
    section='unknown', nen `lenient` KHONG loai — sai so voi qrel=1.
    """
    assert effective_section(d("violated", "unknown", "No prior chemotherapy")) == "exclusion"
    assert effective_section(d("violated", "unknown", "Not pregnant or nursing")) == "exclusion"
    # header that van thang the phu dinh suy ra
    assert effective_section(d("violated", "inclusion", "No prior therapy")) == "inclusion"
    # khong co the phu dinh thi giu nguyen `unknown`
    assert effective_section(d("violated", "unknown", "Histologically confirmed")) == "unknown"

    ca = [d("violated", "unknown", "No prior chemotherapy")] + [d("satisfied")] * 5
    assert trial_score(ca, "lenient") == DISQUALIFIED
    assert trial_score(ca, "lenient", infer_section=False) > DISQUALIFIED, \
        "co ablate duoc suy dien nay — Phase 11 can do no doi bao nhieu ket luan"


def test_unverifiable_khong_bao_gio_loai():
    """Ca diem cua ba trang thai (invariant 1): khong kiem chung duoc thi bi
    xep sau, KHONG bi vut."""
    toan_unv = [d("unverifiable")] * 20
    for rule in RULES:
        assert trial_score(toan_unv, rule) > DISQUALIFIED, rule


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"  HONG  {fn.__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} qua")
    sys.exit(1 if bad else 0)
