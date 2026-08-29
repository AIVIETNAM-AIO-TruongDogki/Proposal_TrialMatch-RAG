"""Phase 4 buoc 1b — goi Ollama qua HTTP, ep dau ra theo JSON Schema.

Dung `requests` (da co san nho pyserini) toi localhost:11434. Khong them phu
thuoc moi, va khong dung thu vien client nao — API cua Ollama chi la mot
endpoint POST, boc them mot lop chi lam mo di dieu gi duoc gui di.

BA THAM SO CO DINH VI DAY LA THI NGHIEM, KHONG PHAI CHATBOT
-----------------------------------------------------------
temperature=0, top_p=1, seed=0. Neu de mac dinh, chay lai cung mot model tren
cung mot benh an se cho ket qua khac nhau, va bang so sanh 6 model o buoc 2 se
do lan nhieu lay mau chu khong phai do chenh lech giua cac model.

Ollama van khong dam bao tai lap tuyet doi (thu tu cong so thuc dau tren GPU),
nen day la giam nhieu chu khong phai xoa bo. Ghi ro dieu do trong bao cao thay
vi mac dinh la da co.
"""

from __future__ import annotations

import json
import time

import requests

HOST = "http://localhost:11434"
TIMEOUT = 300  # model 8B tren 8 GB VRAM co the cham; that bai that thi la loi khac


class OllamaError(RuntimeError):
    pass


def is_up(host: str = HOST) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=5).ok
    except requests.RequestException:
        return False


def list_models(host: str = HOST) -> list[str]:
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []))


def unload(model: str, host: str = HOST) -> None:
    """Tra VRAM ve. Bay 4 cua plan: 8 GB khong du cho ca LLM lan encoder.

    keep_alive=0 bao Ollama nha model ngay thay vi giu 5 phut mac dinh.
    """
    try:
        requests.post(f"{host}/api/generate",
                      json={"model": model, "keep_alive": 0}, timeout=30)
    except requests.RequestException:
        pass


def chat_json(model: str, system: str, user: str, schema: dict,
              host: str = HOST, num_ctx: int = 8192) -> tuple[dict | None, dict]:
    """Mot lan goi co cau truc. Tra ve (du_lieu_da_parse | None, thong_tin_do).

    Tra None khi model tra ve JSON khong hop le — do la mot KET QUA (schema
    validity < 100%), khong phai mot ngoai le. Nem exception o day se lam hong
    ca me benchmark chi vi mot model yeu, va chinh su yeu do la thu ta muon do.
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1, "seed": 0, "num_ctx": num_ctx},
    }

    t0 = time.time()
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise OllamaError(f"{model}: khong goi duoc — {e}") from e
    elapsed = time.time() - t0

    if not r.ok:
        raise OllamaError(f"{model}: HTTP {r.status_code} — {r.text[:200]}")

    body = r.json()
    raw = body.get("message", {}).get("content", "")

    meta = {
        "seconds": round(elapsed, 2),
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
        "raw": raw,
    }

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, meta
    # Model co the tra ve mot mang hoac mot chuoi van hop le JSON nhung sai kieu.
    return (data if isinstance(data, dict) else None), meta
