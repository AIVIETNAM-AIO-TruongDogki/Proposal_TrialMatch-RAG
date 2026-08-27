"""Nap topics va qrels cua TREC Clinical Trials.

Bay so 1 cua bo du lieu nay: CA HAI nam deu danh so topic tu 1. Topic 1 cua
2021 la nam 45 tuoi u sao bao tuy song; topic 1 cua 2022 la nam 19 tuoi tu van
suc khoe tinh duc. Nap chung vao mot dict theo so nguyen tho se AM THAM gop hai
benh nhan khong lien quan lam mot.

Vi vay moi id o day deu duoc dat tien to nam ("2021_1"), va so tran khong bao
gio thoat ra khoi module nay.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

RAWDATA = "rawdata"

# Y nghia nhan, lay nguyen van tu NIST:
#   "Judgment of 0 is non-relevant, 1 is excluded, and 2 is eligible."
# Doc nguoc voi hau het collection TREC khac, nen dat ten thay vi dung so tran.
NOT_RELEVANT = 0
EXCLUDED = 1      # lien quan y khoa NHUNG bi tieu chi loai tru -> ca then chot
ELIGIBLE = 2

DEV_YEAR = 2021   # 75 topic
TEST_YEAR = 2022  # 50 topic — chi cham DUNG MOT LAN, o Phase 11

Qrels = dict[str, dict[str, int]]


def topic_id(year: int, number: str | int) -> str:
    return f"{year}_{number}"


def load_topics(year: int, rawdata: str = RAWDATA) -> dict[str, str]:
    path = os.path.join(rawdata, f"topics{year}.xml")
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for t in root.findall("topic"):
        num = t.get("number")
        if num is None:
            continue
        out[topic_id(year, num)] = " ".join((t.text or "").split())
    return out


def load_qrels(year: int, rawdata: str = RAWDATA) -> Qrels:
    """Doc file qrels dang TREC: `topic 0 nct_id label`."""
    path = os.path.join(rawdata, f"qrels{year}.txt")
    out: Qrels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            tid, _, nct, rel = parts
            out.setdefault(topic_id(year, tid), {})[nct] = int(rel)
    return out


def eligible_only(qrels: Qrels) -> Qrels:
    """Ban do lai nhan cho ho do 'eligibility-aware'.

    ELIGIBLE(2) -> 1,  EXCLUDED(1) -> 0,  NOT_RELEVANT(0) -> 0.

    Day la nua quan trong nhat cua Phase 2. Voi thang do CHINH THUC cua TREC,
    mot trial "lien quan y khoa nhung bi loai tru" duoc gain DUONG — dung cai
    that bai ma ca de tai nay sinh ra de loai bo. Danh gia luan diem chinh bang
    nDCG chinh thuc thoi thi mot bo loc eligibility HOAT DONG TOT se trong nhu
    mot buoc lui. Luon bao cao ca hai ho, canh nhau.
    """
    return {t: {d: (1 if r == ELIGIBLE else 0) for d, r in docs.items()}
            for t, docs in qrels.items()}


def label_counts(qrels: Qrels) -> dict[int, int]:
    out = {NOT_RELEVANT: 0, EXCLUDED: 0, ELIGIBLE: 0}
    for docs in qrels.values():
        for r in docs.values():
            out[r] = out.get(r, 0) + 1
    return out
