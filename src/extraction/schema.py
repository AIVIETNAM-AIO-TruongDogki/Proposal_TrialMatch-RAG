"""Phase 4 buoc 1 — schema ho so benh nhan, va JSON Schema ep Ollama tuan thu.

BA TRANG THAI, KHONG PHAI HAI
-----------------------------
Day la lan dau invariant 2 di vao code: *thong tin benh nhan khong co thi phai
giu nguyen la khong co.* Ba trang thai phai phan biet rach roi:

    stated   benh an CO noi              "severe aortic stenosis"
    negated  benh an noi KHONG co        "no history of diabetes"
    absent   benh an KHONG nhac toi      (truong vang mat hoan toan)

Gop `negated` vao `absent` la loi hay gap nhat va tai hai nhat o phase nay.
"Benh nhan khong bi tieu duong" la mot su that lam sang co the THOA MAN mot tieu
chi loai tru; "benh an khong noi gi ve tieu duong" thi khong thoa man gi ca, no
phai thanh `unverifiable` o Phase 8. Nhap hai cai lam mot la bien mot cau tra
loi dung thanh mot phong doan.

BA QUY UOC MA SCHEMA NAY AP DAT
-------------------------------
1. `status` CHI co `present` | `negated`. Khong co gia tri "absent".
   Do la co y: khong the trich dan bang chung cho mot thu khong duoc nhac toi.
   `absent` duoc bieu dat bang cach muc do KHONG NAM trong danh sach.

2. Moi gia tri trich duoc deu BAT BUOC co `evidence`. Khong co truong nao mien.
   Do la thu bien invariant 3 tu loi hua thanh phep do — xem verify.py.

3. KHONG cho phep `null`. Neu cho, ta se nhan duoc ca {"age": null} lan
   {"age": {"value": null}} va mat kha nang phan biet "da doc va khong thay"
   voi "chua doc". Vang mat la cach duy nhat de noi "khong co".

`age` va `sex` co y KHONG nam trong `required` o cap goc — mot benh an that su
co the khong noi tuoi, va model phai duoc phep im lang thay vi doan.
"""

from __future__ import annotations

# Nhom thuc the trich xuat. Thu tu nay dung lai o query.py va o bang bao cao.
LIST_FIELDS = (
    "conditions",        # chan doan, benh chinh
    "biomarkers",        # EGFR, HER2, PD-L1, dot bien...
    "prior_treatments",  # thuoc / phau thuat / xa tri da nhan
    "labs",              # ket qua xet nghiem co gia tri
    "comorbidities",     # benh kem
)

SCALAR_FIELDS = ("age", "sex")

# Truong duoc phep dua vao truy van BM25 o Phase 4 buoc 4.
# `labs` bi loai: gia tri so ("WBC 11.2") khong phai term truy hoi tot, va
# `comorbidities` giu lai vi ten benh kem van la tin hieu chu de.
QUERY_FIELDS = ("conditions", "biomarkers", "prior_treatments", "comorbidities")

_EVIDENCE = {
    "type": "string",
    # Do tren model 3B: no tra ve evidence="" ma van hop schema, roi verify.py
    # phai vut bo mot gia tri dung. Bat do dai toi thieu ngay tu schema.
    "minLength": 3,
    "description": "Trich NGUYEN VAN tu benh an. Phai la chuoi con dung tung "
                   "chu cua benh an goc, khong dien dat lai, khong tom tat.",
}

_ENTITY = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["present", "negated"],
            "description": "present = benh an noi CO. negated = benh an noi "
                           "KHONG co. Neu benh an khong nhac toi, DUNG dua "
                           "muc nay vao danh sach.",
        },
        "evidence": _EVIDENCE,
    },
    "required": ["name", "status", "evidence"],
    "additionalProperties": False,
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string", "enum": ["years", "months", "weeks", "days"]},
                "evidence": _EVIDENCE,
            },
            "required": ["value", "unit", "evidence"],
            "additionalProperties": False,
        },
        "sex": {
            "type": "object",
            "properties": {
                "value": {"type": "string", "enum": ["male", "female"]},
                "evidence": _EVIDENCE,
            },
            "required": ["value", "evidence"],
            "additionalProperties": False,
        },
        **{f: {"type": "array", "items": _ENTITY} for f in LIST_FIELDS},
    },
    # Chi bat buoc cac danh sach (co the rong). age/sex duoc phep vang mat —
    # do la cach model noi "benh an khong cho biet", va no la mot cau tra loi
    # HOP LE, khong phai that bai.
    "required": list(LIST_FIELDS),
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract structured clinical facts from a patient narrative for clinical \
trial matching. You are a careful medical information extractor, not a \
diagnostician.

Rules, in order of importance:

1. NEVER infer, impute, or guess. Extract only what the narrative literally \
states. If the narrative does not mention something, leave it out entirely. \
Omission is a correct answer; guessing is not.

2. Every extracted value MUST carry an `evidence` field containing a VERBATIM \
substring of the narrative. Copy the exact characters. Do not paraphrase, \
normalize, expand abbreviations, or fix typos inside `evidence`. If you cannot \
quote it exactly, do not extract it.

3. Distinguish two different things:
   - status "present": the narrative says the patient HAS it.
   - status "negated": the narrative says the patient does NOT have it \
("no history of diabetes", "denies chest pain", "ruled out for sepsis").
   A negation is a clinical fact worth recording. Record it as "negated" with \
the negating phrase as evidence. Never record it as "present", and never drop \
it silently.

4. If the narrative does not mention a condition at all, it does not belong in \
the output in any form.

5. `name` should be the clinical concept in its normal form (e.g. "type 2 \
diabetes"). Only `evidence` must be verbatim.
"""

USER_TEMPLATE = "Patient narrative:\n\n{narrative}\n\nExtract the structured profile."


def prompt_hash() -> str:
    """Van tay cua prompt + schema, dung lam mot phan khoa cache.

    Doi prompt hay doi schema thi cache phai hong — neu khong ta se so sanh
    ket qua cua hai cau hoi khac nhau va tuong la so sanh hai model.
    """
    import hashlib
    import json

    blob = SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(PROFILE_SCHEMA, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
