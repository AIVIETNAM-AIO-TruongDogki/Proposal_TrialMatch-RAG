"""Phase 8 buoc 1 — schema quyet dinh muc TIEU CHI, ba trang thai.

BA TRANG THAI LA DONG GOP, KHONG PHAI CHI TIET KY THUAT
--------------------------------------------------------
    satisfied     benh an noi dieu THOA MAN tieu chi
    violated      benh an noi dieu KHONG THOA MAN tieu chi
    unverifiable  benh an KHONG NOI GI ve dieu tieu chi hoi

Voi tieu chi LOAI TRU, huong bi dao: "satisfied" nghia la benh nhan KHONG co
dieu bi loai tru (nen khong bi loai), "violated" nghia la CO (nen bi loai).
Prompt noi ro dieu nay vi day la cho de nham nhat.

Gop `unverifiable` vao mot trong hai trang thai kia se pha huy chinh dong gop
cua de tai (invariant 1). Vi vay co `FORCED_SCHEMA` — ban BO `unverifiable` —
de chay ablation lua chon ep buoc: neu hai trang thai lam cung tot thi dong
gop ba trang thai CHUA duoc chung minh, va nguoi phan bien se hoi dieu do.

HAI TRICH DAN, KHONG PHAI MOT
------------------------------
`criterion_quote` phai la chuoi con nguyen van cua TIEU CHI (kiem bang
store.verify_quote, doi chieu span offset vao criteria_raw goc).
`patient_evidence` phai la chuoi con nguyen van cua BENH AN.

Hai phia phai kiem rieng: mot model co the trich dung tieu chi roi bia bang
chung benh nhan, hoac nguoc lai. Chi kiem mot phia la de lot nua so loi.
"""

from __future__ import annotations

import hashlib
import json

import prompts

LABELS = ("satisfied", "violated", "unverifiable")
FORCED_LABELS = ("satisfied", "violated")   # ablation lua chon ep buoc

_DECISION_PROPS = {
    "label": {
        "type": "string",
        "enum": list(LABELS),
        "description": "unverifiable la MAC DINH khi benh an khong noi gi.",
    },
    "criterion_quote": {
        "type": "string",
        "minLength": 3,
        "description": "Trich NGUYEN VAN tu van ban tieu chi, khong dien dat lai.",
    },
    "patient_evidence": {
        "type": "string",
        "description": "Trich NGUYEN VAN tu benh an. De RONG neu unverifiable.",
    },
    "reasoning": {"type": "string", "maxLength": 300},
}


def decision_schema(forced: bool = False) -> dict:
    """Schema mot quyet dinh (che do goi tung tieu chi)."""
    props = json.loads(json.dumps(_DECISION_PROPS))
    if forced:
        props["label"]["enum"] = list(FORCED_LABELS)
        props["label"]["description"] = ("CHI hai lua chon — bat buoc chon mot "
                                          "ben ke ca khi benh an khong du thong tin.")
    return {"type": "object", "properties": props,
            "required": ["label", "criterion_quote", "patient_evidence"],
            "additionalProperties": False}


def batch_schema(n: int, forced: bool = False) -> dict:
    """Schema goi CA TRIAL mot lan — moi muc mang `criterion_idx` de khop lai.

    Khop theo idx chu khong theo thu tu mang: mot lan khop sai se gan quyet
    dinh cua tieu chi nay cho tieu chi khac, dung dieu invariant 3 cam.
    """
    item = decision_schema(forced)
    item["properties"] = {"criterion_idx": {"type": "integer"}, **item["properties"]}
    item["required"] = ["criterion_idx", *item["required"]]
    return {"type": "object",
            "properties": {"decisions": {"type": "array", "items": item,
                                          "minItems": n, "maxItems": n}},
            "required": ["decisions"], "additionalProperties": False}


SYSTEM = prompts.load("eligibility_system")
BATCH_SYSTEM = SYSTEM + "\n\n" + prompts.load("eligibility_batch_addendum")


def user_prompt(narrative: str, nct_id: str, section: str, criterion: str) -> str:
    return prompts.load("eligibility_user").format(
        narrative=narrative, nct_id=nct_id, section=section, criterion=criterion)


def batch_user_prompt(narrative: str, nct_id: str, criteria: list[dict]) -> str:
    tpl = prompts.load("eligibility_batch_item")
    body = "\n".join(tpl.format(idx=c["idx"], section=c["section"], text=c["text"])
                     for c in criteria)
    return prompts.load("eligibility_batch_user").format(
        narrative=narrative, nct_id=nct_id, n=len(criteria), criteria=body)


def prompt_hash(forced: bool = False, batched: bool = True) -> str:
    """Van tay prompt+schema, mot phan khoa cache — doi prompt thi cache phai hong."""
    blob = ((BATCH_SYSTEM if batched else SYSTEM)
            + prompts.load("eligibility_batch_item")
            + prompts.load("eligibility_batch_user" if batched else "eligibility_user")
            + json.dumps(decision_schema(forced), sort_keys=True))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
