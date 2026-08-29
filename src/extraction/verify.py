"""Phase 4 buoc 2 — kiem chung co hoc. Day la chot chan quan trong nhat.

Model 3-8B chay cuc bo se bia. Khong ngan duoc. Nhung PHAT HIEN va VUT BO duoc
phan bia, va ty le bi vut chinh la mot con so bao cao duoc.

Nguyen tac: moi gia tri trich ra deu mang mot `evidence`, va `evidence` phai la
CHUOI CON nguyen van cua benh an. So khop sau khi chuan hoa khoang trang va ha
chu thuong — dung khuon mau cua store.verify_quote(), cung mot triet ly: bien
mot loi hua thanh mot phep do.

NHAN VANG MIEN PHI
------------------
Tuoi va gioi gan nhu luon nam o cau dau benh an. Regex duoi day da do tren ca
75 topic dev: **tuoi 75/75, gioi 74/75**. Truong hop trot duy nhat la topic
2021_14 ("70 y/o with COPD...") — benh an that su KHONG noi gioi tinh.

Vi vay 2021_14 khong phai lo hong cua regex; no la mot BAY co gia tri:
model nao xuat ra gioi tinh cho benh an nay la dang bia, va ta bat duoc dieu do
ma khong ton mot dong gan nhan tay nao. Xem FABRICATION_TRAPS.

PHEP DO NAY KHONG BAT DUOC GI — DOC TRUOC KHI TIN VAO CON SO
------------------------------------------------------------
Kiem tra chuoi con xac nhan rang EVIDENCE co that trong benh an. No KHONG xac
nhan rang `name` suy ra duoc tu evidence do. Mot model hoan toan co the trich
dan dung nguyen van roi gan cho no mot nhan sai.

Da thay dung truong hop nay o model 3B, topic 2021_2:

    name:     "hypertrophic cardiomyopathy"
    evidence: "left ventricular hypertrophy with cavity dilation and severe
               global hypokinesis"          <- co that trong benh an

Benh an KHONG he noi benh co tim phi dai. Model da chan doan thay vi trich
xuat — dung dieu invariant 2 cam — va phep kiem tra co hoc cho no di qua.

Vi vay `grounding` la dieu kien CAN, khong phai dieu kien DU. Hand-audit 25
benh an o buoc 5 ton tai chinh xac vi lo hong nay, va no khong the tu dong hoa
bang du lieu hien co. Dung bao cao grounding nhu the no do duoc tinh dung.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from src.eval import data
from src.extraction.schema import LIST_FIELDS, SCALAR_FIELDS

PROFILE_DIR = "data/profiles"

# --- Nhan vang tu regex (da do: tuoi 75/75, gioi 74/75 tren dev) --------------

_UNIT = r"(day|week|month|year)"
_AGE_PATS = [
    re.compile(rf"\b(\d{{1,3}})\s*[-]?\s*{_UNIT}s?\s*[-]?\s*old", re.I),
    re.compile(rf"\b(\d{{1,3}})\s*{_UNIT}s?\s+(?=man|woman|male|female)", re.I),
    re.compile(r"\b(\d{1,3})\s*[-]?\s*(?:y/o|yo)\b", re.I),
    re.compile(r"\b(\d{1,3})\s*[MF]\b"),
]
_SEX_AB = re.compile(r"\b\d{1,3}\s*[-]?\s*(?:y/?o|years?\s*[-]?\s*old)?\s*([MF])\b")
_SEX_M = re.compile(r"\b(?:man|male|gentleman|boy)\b", re.I)
_SEX_F = re.compile(r"\b(?:woman|female|lady|girl)\b", re.I)
_PRO_M = re.compile(r"\b(?:he|his|him)\b", re.I)
_PRO_F = re.compile(r"\b(?:she|her|hers)\b", re.I)

# Benh an KHONG noi gioi tinh. Model nao dien vao la dang bia.
FABRICATION_TRAPS = {"2021_14": "sex"}

# Dau hieu phu dinh — dung do xem model co bat duoc `negated` khong.
NEG_CUES = re.compile(
    r"\b(?:no\s+(?:history|evidence|signs?|prior)|denies|without|"
    r"negative\s+for|ruled\s+out|not\s+(?:on|taking)|free\s+of|absence\s+of)\b",
    re.I)


def gold_age_sex(text: str) -> tuple[int | None, str | None, str | None]:
    """(tuoi, don_vi, gioi) suy tu regex. None = benh an khong noi."""
    head = text[:220]
    age = unit = None
    for p in _AGE_PATS:
        m = p.search(head)
        if m:
            age = int(m.group(1))
            unit = (m.group(2).lower() + "s") if (m.lastindex or 0) >= 2 else "years"
            break

    sex = None
    m = _SEX_AB.search(head)
    if m:
        sex = "male" if m.group(1) == "M" else "female"
    else:
        fm, ff = _SEX_M.search(head), _SEX_F.search(head)
        if fm or ff:
            sex = ("male" if (ff is None or (fm and fm.start() < ff.start()))
                   else "female")
        else:
            pm, pf = _PRO_M.search(head), _PRO_F.search(head)
            if pm or pf:
                sex = ("male" if (pf is None or (pm and pm.start() < pf.start()))
                       else "female")
    return age, unit, sex


# --- Kiem chung -------------------------------------------------------------

def norm(s: str) -> str:
    """Chuan hoa y het store.verify_quote(): gop khoang trang, ha chu thuong.

    Benh an goc co the xuong dong giua chung con model tra ve mot dong lien;
    so khop tho se bao sai cho mot trich dan hoan toan dung.
    """
    return " ".join(s.split()).lower()


# Do dai toi da cua mot trich dan con duoc coi la "chi ra duoc cho nao".
# Benh an trung binh 135 tu; mot evidence 60 tu khong dinh vi duoc gi ca.
MAX_EVIDENCE_WORDS = 30


def grounded(evidence: str, narrative: str) -> bool:
    e = norm(evidence)
    return bool(e) and e in norm(narrative)


def localized(evidence: str) -> bool:
    """Trich dan co du NGAN de thuc su la bang chung khong?

    `grounded` mot minh la thuoc do CO THE LACH: model trich nguyen ca benh an
    cho moi truong se dat 100% grounding ma khong chi ra duoc gi. Da thay dung
    hien tuong do o model 3B. Bao cao hai cot canh nhau thay vi gop lam mot.
    """
    return len(evidence.split()) <= MAX_EVIDENCE_WORDS


def schema_ok(profile: dict) -> bool:
    """Kiem tra thu cong thay vi keo them phu thuoc `jsonschema`.

    Chi kiem nhung gi thuc su rang buoc y nghia: danh sach bat buoc ton tai,
    enum dung, va MOI muc deu co evidence.
    """
    if not isinstance(profile, dict):
        return False
    for f in LIST_FIELDS:
        v = profile.get(f)
        if not isinstance(v, list):
            return False
        for it in v:
            if not isinstance(it, dict):
                return False
            if it.get("status") not in ("present", "negated"):
                return False
            if not isinstance(it.get("name"), str) or not isinstance(it.get("evidence"), str):
                return False
    if "age" in profile:
        a = profile["age"]
        if not isinstance(a, dict) or not isinstance(a.get("value"), (int, float)):
            return False
        if not isinstance(a.get("evidence"), str):
            return False
    if "sex" in profile:
        s = profile["sex"]
        if not isinstance(s, dict) or s.get("value") not in ("male", "female"):
            return False
        if not isinstance(s.get("evidence"), str):
            return False
    return True


def verify_profile(profile: dict, narrative: str) -> tuple[dict, list[dict]]:
    """Vut bo moi truong co evidence khong phai chuoi con that.

    Tra ve (ho_so_sach, danh_sach_bi_loai). Danh sach bi loai duoc ghi ra dia
    de hand-audit o buoc 5 xem duoc model da bia CAI GI, khong chi bao nhieu.
    """
    clean: dict = {}
    dropped: list[dict] = []

    for f in SCALAR_FIELDS:
        v = profile.get(f)
        if not isinstance(v, dict):
            continue
        if grounded(v.get("evidence", ""), narrative):
            clean[f] = v
        else:
            dropped.append({"field": f, "value": v.get("value"),
                            "evidence": v.get("evidence", "")})

    for f in LIST_FIELDS:
        keep = []
        for it in profile.get(f) or []:
            if not isinstance(it, dict):
                continue
            if grounded(it.get("evidence", ""), narrative):
                keep.append(it)
            else:
                dropped.append({"field": f, "value": it.get("name"),
                                "evidence": it.get("evidence", "")})
        clean[f] = keep

    return clean, dropped


def n_values(profile: dict) -> int:
    return (sum(1 for f in SCALAR_FIELDS if f in profile)
            + sum(len(profile.get(f) or []) for f in LIST_FIELDS))


# --- Cham diem mot model ----------------------------------------------------

def score_model(model: str, year: int, topics: dict[str, str],
                profile_dir: str = PROFILE_DIR) -> dict | None:
    path = os.path.join(profile_dir, f"{year}.{model.replace(':', '_')}.json")
    if not os.path.exists(path):
        return None
    recs = json.load(open(path, encoding="utf-8"))

    n = len(recs)
    valid = tot_vals = ok_vals = loc_vals = 0
    age_hit = age_tot = sex_hit = sex_tot = 0
    fabricated = 0
    neg_topics = neg_hit = 0
    secs: list[float] = []

    for tid, rec in recs.items():
        narrative = topics[tid]
        secs.append(rec.get("seconds", 0.0))
        prof = rec.get("profile")

        if prof is None or not schema_ok(prof):
            continue
        valid += 1

        # Grounding tinh tren ho so THO, truoc khi loc.
        raw_n = n_values(prof)
        clean, dropped = verify_profile(prof, narrative)
        tot_vals += raw_n
        ok_vals += raw_n - len(dropped)
        loc_vals += sum(
            1 for f in SCALAR_FIELDS if f in clean and localized(clean[f]["evidence"])
        ) + sum(
            1 for f in LIST_FIELDS for it in (clean.get(f) or [])
            if localized(it["evidence"]))

        # Tuoi/gioi so voi nhan vang — tinh tren ho so DA LOC, vi do moi la
        # thu he thong that su dung.
        g_age, g_unit, g_sex = gold_age_sex(narrative)
        if g_age is not None:
            age_tot += 1
            a = clean.get("age") or {}
            if a.get("value") == g_age and (a.get("unit") or "years") == g_unit:
                age_hit += 1
        if g_sex is not None:
            sex_tot += 1
            if (clean.get("sex") or {}).get("value") == g_sex:
                sex_hit += 1

        # Bay bia dat: benh an khong noi, model van dien.
        trap = FABRICATION_TRAPS.get(tid)
        if trap and trap in clean:
            fabricated += 1

        # Co bat duoc phu dinh khong?
        if NEG_CUES.search(narrative):
            neg_topics += 1
            if any(it.get("status") == "negated"
                   for f in LIST_FIELDS for it in (clean.get(f) or [])):
                neg_hit += 1

    pct = lambda a, b: (a / b * 100) if b else float("nan")
    return {
        "model": model,
        "n": n,
        "schema_valid": pct(valid, n),
        "grounding": pct(ok_vals, tot_vals),
        "localized": pct(loc_vals, ok_vals),
        "age_acc": pct(age_hit, age_tot),
        "sex_acc": pct(sex_hit, sex_tot),
        "coverage": (ok_vals / valid) if valid else 0.0,
        "neg_recall": pct(neg_hit, neg_topics),
        "fabricated": fabricated,
        "sec_per_call": (sum(secs) / len(secs)) if secs else 0.0,
    }


# --- CLI --------------------------------------------------------------------

# So lan goi Phase 8 can cho tap dev, do tren runs/bm25_best.dev.txt:
# 75 topic x top-20 trial x 18,0 tieu chi/trial.
PHASE8_CALLS = 27045


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=data.DEV_YEAR)
    ap.add_argument("--profile-dir", default=PROFILE_DIR)
    args = ap.parse_args()

    topics = data.load_topics(args.year)
    if not os.path.isdir(args.profile_dir):
        print(f"Chua co {args.profile_dir}. Chay src.extraction.extract truoc.",
              file=sys.stderr)
        return 1

    models = sorted({f.split(".", 2)[1].replace("_", ":", 1)
                     for f in os.listdir(args.profile_dir)
                     if f.startswith(f"{args.year}.") and f.endswith(".json")})
    rows = [r for m in models if (r := score_model(m, args.year, topics, args.profile_dir))]
    if not rows:
        print("Khong co ket qua nao.", file=sys.stderr)
        return 1

    hdr = (f"{'model':24s} {'schema':>7s} {'ground':>7s} {'local':>7s} "
           f"{'age':>6s} {'sex':>6s} {'cover':>6s} {'neg':>6s} {'bia':>4s} "
           f"{'s/goi':>7s} {'P8':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["grounding"]):
        p8 = r["sec_per_call"] * PHASE8_CALLS / 3600
        print(f"{r['model']:24s} {r['schema_valid']:6.1f}% {r['grounding']:6.1f}% "
              f"{r['localized']:6.1f}% "
              f"{r['age_acc']:5.1f}% {r['sex_acc']:5.1f}% {r['coverage']:6.1f} "
              f"{r['neg_recall']:5.1f}% {r['fabricated']:4d} "
              f"{r['sec_per_call']:6.2f}s {p8:7.1f}h")

    print(f"\nP8 = du phong gio GPU cho Phase 8 tren tap dev ({PHASE8_CALLS:,} lan goi,")
    print("     moi tieu chi mot lan goi). Gop ca trial vao mot lan goi thi chia ~18.")
    print("bia  = so benh an KHONG noi thong tin do ma model van dien "
          f"(bay: {', '.join(FABRICATION_TRAPS)}).")

    out = os.path.join("results", f"_extract_bench.{args.year}.json")
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nDa ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
