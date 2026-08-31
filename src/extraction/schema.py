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

import prompts

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

# Noi dung prompt song trong prompts/*.txt, khong phai hang so Python — sua
# prompt khong con dong nghia voi sua code, va diff tren prompt la diff dung
# tren prompt.
SYSTEM_PROMPT = prompts.load("extraction_system")
USER_TEMPLATE = prompts.load("extraction_user")

# --- Goi theo lo (batch) -----------------------------------------------------
#
# extract.py goi Gemini mot lan cho N benh an thay vi N lan rieng, de do so
# request khi dung free tier (5 req/phut, 20 req/ngay — xem
# docs/decisions/phase4-gemini-backend.md). N benh an DOC LAP voi nhau; addendum
# duoi day noi ro dieu do va cam tron thong tin giua cac benh an.
#
# Moi phan tu tra ve mang them `index` de khop lai DUNG benh an ban dau — khong
# dua vao thu tu mang tra ve, vi mot loi khop sai se gan ho so cua benh nhan
# nay cho benh nhan khac, dung dieu invariant 3 (moi ket luan phai co can cu
# dung nguon) cam.
BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + prompts.load("extraction_batch_addendum")

_BATCH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {"index": {"type": "integer"}, **PROFILE_SCHEMA["properties"]},
    "required": ["index", *PROFILE_SCHEMA["required"]],
    "additionalProperties": False,
}


def batch_schema(n: int) -> dict:
    """Schema cho mot lan goi gom n ho so — PROFILE_SCHEMA cong truong `index`."""
    return {
        "type": "object",
        "properties": {
            "profiles": {"type": "array", "items": _BATCH_ITEM_SCHEMA,
                        "minItems": n, "maxItems": n},
        },
        "required": ["profiles"],
        "additionalProperties": False,
    }


def batch_user_prompt(narratives: list[str]) -> str:
    header = prompts.load("extraction_batch_header").format(n=len(narratives))
    item_tpl = prompts.load("extraction_batch_item")
    items = "\n\n".join(item_tpl.format(index=i, narrative=narr)
                        for i, narr in enumerate(narratives))
    return f"{header}\n\n{items}"


def prompt_hash() -> str:
    """Van tay cua prompt + schema, dung lam mot phan khoa cache.

    Doi prompt hay doi schema thi cache phai hong — neu khong ta se so sanh
    ket qua cua hai cau hoi khac nhau va tuong la so sanh hai model. Hash tren
    BATCH_SYSTEM_PROMPT (da bao gom SYSTEM_PROMPT ben trong) vi do la thu
    extract.py thuc su gui di.
    """
    import hashlib
    import json

    blob = (BATCH_SYSTEM_PROMPT + prompts.load("extraction_batch_header") +
            prompts.load("extraction_batch_item") +
            json.dumps(PROFILE_SCHEMA, sort_keys=True))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
