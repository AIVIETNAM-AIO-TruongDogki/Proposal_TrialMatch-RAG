"""Phase 4 buoc 1b — goi Gemini API, xoay vong 3 key, ep dau ra theo JSON Schema.

Thay the ollama.py lam backend chinh (quyet dinh nguoc lai "fully local, no
API" — xem docs/decisions/phase4-gemini-backend.md). Cung chu ky chat_json()
nhu ollama.py de extract.py/query.py doi backend chi bang mot dong import.

XOAY VONG KEY — MOI LAN GOI, KHONG PHAI CHI KHI LOI
----------------------------------------------------
Key ke tiep duoc dung cho MOI lan goi, bat ke lan truoc thanh cong hay khong —
co y de trai deu tai qua ca 3 quota rieng cua tung key (RPM/RPD free tier),
khong phai chi la du phong khi loi.

Khi mot lan goi gap loi TAM THOI (429 rate-limit, 5xx), ham thu lai voi key KE
TIEP tu chinh vong xoay dang chay, toi da len(KEYS) lan roi moi nem loi. Loi
KHONG tam thoi (auth sai, doi so sai) nem ngay — thu lai qua key khac chi ton
quota vo ich cho mot loi se lap lai giong het.

MODEL — DA KIEM CHUNG THUC TE, KHONG DOAN
------------------------------------------
`gemini-2.5-flash` (lua chon ban dau) tra ve 404 "no longer available to new
users" cho tai khoan nay (kiem tra thang 8/2026) — API tu goi y
`gemini-3.6-flash`, da thu va hoat dong dung nhu mong doi voi PROFILE_SCHEMA
day du (xem docs/decisions/phase4-gemini-backend.md).

response_schema cua Gemini la mot tap con OpenAPI, KHONG phai JSON Schema day
du — da kiem tra thuc te: `additionalProperties` bi Gemini tu choi thang voi
loi 400, con `minLength`, `enum`, `description`, `required` (ke ca long nhieu
cap) deu duoc chap nhan nguyen ven. `_to_gemini_schema()` chi loc mai
`additionalProperties`, khong dong gi khac.

thinking_level=MINIMAL vi day la trich xuat co cau truc, khong phai suy luan
mo — de thinking mac dinh ton hang tram token "nghi" vo ich cho moi lan goi.
"""

from __future__ import annotations

import itertools
import json
import os
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Da do bang thuc nghiem tren ca 75 topic dev (bang so sanh: docs/phase4-tong-ket.md
# muc 6.5). `gemini-3.6-flash` trich xuat tot hon o do phu (16,9 vs 13,0 gia tri/benh
# an) va nhat la o PHU DINH (bat 92% vs 58% so phu dinh that) — nhung Lite nhanh gap
# 2,8 lan va co han ngach RIENG, thu quyet dinh kha thi cua Phase 8 (27.045 lan goi).
# Chon Lite la mot DANH DOI CO Y THUC, khong phai vi hai model tuong duong:
# recall phu dinh thap se lam yeu buoc ket luan `satisfied` cho tieu chi loai tru o
# Phase 8, va dieu do phai duoc do lai o day chu khong duoc quen.
MODEL = "gemini-3.5-flash-lite"
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class GeminiError(RuntimeError):
    pass


def _load_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    return keys


KEYS = _load_keys()
_cycle = itertools.cycle(KEYS) if KEYS else None


def _retry_delay_seconds(e: errors.APIError, default: float = 2.0, cap: float = 20.0) -> float:
    """Doc goi y `retryDelay` cua chinh Gemini trong loi 429/5xx, thay vi doan.

    Quan sat thuc te: `generate_content_free_tier_requests` la mot quota
    RPM chung cho ca 3 key neu chung cung mot du an Google — thu lai NGAY
    voi key ke tiep khi dinh 429 se that bai lien tuc trong cung mot cua so
    phut. Cho theo dung retryDelay ma server tra ve moi co y nghia.
    """
    try:
        for d in e.details.get("error", {}).get("details", []):
            if str(d.get("@type", "")).endswith("RetryInfo"):
                return min(float(str(d["retryDelay"]).rstrip("s")), cap)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return default


def _to_gemini_schema(node):
    """Loc `additionalProperties` khoi cay schema — xem docstring module."""
    if isinstance(node, dict):
        return {k: _to_gemini_schema(v) for k, v in node.items()
                if k != "additionalProperties"}
    if isinstance(node, list):
        return [_to_gemini_schema(v) for v in node]
    return node


def chat_json(model: str, system: str, user: str, schema: dict) -> tuple[dict | None, dict]:
    """Mot lan goi co cau truc. Tra ve (du_lieu_da_parse | None, thong_tin_do).

    Cung quy uoc voi ollama.chat_json: JSON khong hop le tra None, khong nem
    ngoai le — do la mot KET QUA duoc do (ty le schema-valid), khong phai loi
    he thong lam hong ca me benchmark.
    """
    if not KEYS:
        raise GeminiError(
            "Khong co GEMINI_API_KEY_1/2/3 nao trong .env. Xem .env.example.")

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=_to_gemini_schema(schema),
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )

    last_err: Exception | None = None
    for _ in range(len(KEYS)):
        key = next(_cycle)
        client = genai.Client(api_key=key)  # giu tham chieu song — chain truc
        t0 = time.time()                    # tiep lam Client bi GC giua chung
        try:                                 # loi "client has been closed"
            resp = client.models.generate_content(
                model=model, contents=user, config=config)
        except errors.APIError as e:
            last_err = e
            if e.code in _RETRYABLE_CODES:
                time.sleep(_retry_delay_seconds(e))
                continue  # vong xoay da tu chuyen sang key ke tiep
            raise GeminiError(f"{model}: {e}") from e
        except httpx.TransportError as e:
            # Loi tang MANG (mat DNS, mat ket noi, timeout) — xay ra TRUOC KHI
            # toi duoc server Google, nen KHONG phai errors.APIError va SDK
            # khong boc lai no. Quan sat thuc te (03-04/09/2026): mat DNS
            # thoang qua giua dem lam Phase 8 --forced CHET HAN — ngoai le nem
            # thang ra ngoai ham nay, khong bi GeminiError bat, lam ca tien
            # trinh crash cho toi khi nguoi dung tu phat hien. Coi day la TAM
            # THOI giong 429/5xx: khong co retryDelay de doc (khong co response
            # nao ca) nen cho mot khoang co dinh roi thu key ke tiep.
            last_err = e
            time.sleep(2.0)
            continue
        elapsed = time.time() - t0

        usage = resp.usage_metadata
        meta = {
            "seconds": round(elapsed, 2),
            "prompt_tokens": usage.prompt_token_count if usage else None,
            "output_tokens": usage.candidates_token_count if usage else None,
            "raw": resp.text,
        }
        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            return None, meta
        return (data if isinstance(data, dict) else None), meta

    raise GeminiError(f"{model}: het {len(KEYS)} key deu loi tam thoi, "
                       f"loi cuoi: {last_err}")


def self_test() -> bool:
    """Kiem tra tung key auth duoc va schema con nguyen sau khi loc.

    Chay bang tay mot lan truoc dot trich xuat dau tien:
        python -m src.extraction.gemini --self-test
    Khong tu dong chay trong extract.py vi day la mot loi goi API that co that
    (chi phi + do tre), khong phai kiem tra ket noi cuc bo re nhu is_up() cua
    Ollama.
    """
    probe_schema = {"type": "object",
                     "properties": {"ok": {"type": "boolean"}},
                     "required": ["ok"], "additionalProperties": False}
    all_ok = True
    for i in range(1, len(KEYS) + 1):
        try:
            data, meta = chat_json(MODEL, "Reply with ok=true.",
                                    "Confirm you are working.", probe_schema)
            ok = isinstance(data, dict) and data.get("ok") is True
            print(f"key {i}: {'OK' if ok else 'JSON khong nhu mong doi'} "
                  f"({meta['seconds']}s) {data}")
            all_ok &= ok
        except GeminiError as e:
            print(f"key {i}: LOI {e}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(f"{len(KEYS)} GEMINI_API_KEY_* da nap tu .env. "
          f"Dung: python -m src.extraction.gemini --self-test")
