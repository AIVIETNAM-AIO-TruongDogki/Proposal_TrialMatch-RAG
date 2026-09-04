"""Phase 4 buoc 3 — chay trich xuat theo lo (batch), co cache.

    python -m src.extraction.extract --year 2021 --model gemini-3.6-flash

Ghi ra data/profiles/{year}.{model}.json. Moi ban ghi giu CA HAI phien ban:
`profile` tho tu model va `clean` sau khi loc grounding, kem `dropped` — danh
sach nhung gi bi vut vi khong trich dan duoc.

Giu lai phan bi vut la co y. "Ty le grounding 91%" chi la mot con so; danh sach
9% bi vut moi cho biet model bia KIEU GI, va do la thu can cho hand-audit o
buoc 5. Neu chi ghi ban da loc thi ta xoa mat bang chung cua chinh phep do.

GOI THEO LO
-----------
Moi lan goi Gemini xu ly `--batch-size` benh an cung luc (mac dinh 5) thay vi
tung benh an mot — giam so request tren free tier (5/phut, 20/ngay, xem
docs/decisions/phase4-gemini-backend.md). Model tra ve moi ho so kem `index`
de khop lai DUNG benh an; index thieu, trung, hoac ngoai khoang chi lam HONG
benh an do, khong lam hong ca lo (xem schema.batch_schema).

`seconds` ghi trong ban ghi la thoi gian CA LO chia deu cho so benh an trong
lo, de trung binh "s/goi" cua verify.py van doc duoc nhu "giay moi benh an".
`prompt_tokens`/`output_tokens` giu nguyen TONG CA LO — chia deu se sai vi
phan lon token he thong (system prompt) chi ton mot lan cho ca lo, khong
phai N lan.

CACHE
-----
Khoa cache gom (year, model, prompt_hash). Doi prompt hay doi schema thi
prompt_hash doi va cache tu hong dung cho — khong con nguy co so ket qua cua
hai cau hoi khac nhau roi tuong la so hai model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.eval import data
from src.extraction import gemini, schema, verify

PROFILE_DIR = "data/profiles"
BATCH_SIZE = 5


def load_cache(path: str, ph: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        blob = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if blob.get("prompt_hash") != ph:
        print(f"  prompt/schema da doi ({blob.get('prompt_hash')} -> {ph}); bo cache cu")
        return {}
    return blob.get("records", {})


def _save(path: str, ph: str, model: str, recs: dict) -> None:
    json.dump({"prompt_hash": ph, "model": model, "records": recs},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def extract_one(narrative: str, model: str = gemini.MODEL
                ) -> tuple[dict | None, list[dict], dict]:
    """Trich xuat MOT benh an moi tai thoi diem request — khong dinh vao topic
    nam nao, khong dinh vao cache tren dia.

    Tra ve (clean, dropped, meta). `clean` la ho so da qua loc grounding, `None`
    neu model tra sai schema. Tai dung dung nguyen `schema.batch_schema(1)` /
    `schema.batch_user_prompt([narrative])` — schema lo da nhan mang do dai bat
    ky, mot phan tu chi la truong hop n=1, khong can schema rieng.
    """
    out, meta = gemini.chat_json(model, schema.BATCH_SYSTEM_PROMPT,
                                 schema.batch_user_prompt([narrative]),
                                 schema.batch_schema(1))
    items = (out or {}).get("profiles") or []
    it = next((x for x in items if isinstance(x, dict) and x.get("index") == 0), None)
    if it is None:
        return None, [], meta
    prof = {k: v for k, v in it.items() if k != "index"}
    if not verify.schema_ok(prof):
        return None, [], meta
    clean, dropped = verify.verify_profile(prof, narrative)
    return clean, dropped, meta


def run(model: str, year: int, limit: int | None = None,
        profile_dir: str = PROFILE_DIR, force: bool = False,
        batch_size: int = BATCH_SIZE) -> int:
    topics = data.load_topics(year)
    if limit:
        topics = dict(list(topics.items())[:limit])

    ph = schema.prompt_hash()
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, f"{year}.{model.replace(':', '_')}.json")
    recs = {} if force else load_cache(path, ph)

    todo = [t for t in topics if t not in recs]
    n_batches = -(-len(todo) // batch_size) if todo else 0
    print(f"{model}: {len(topics)} benh an, {len(recs)} da co cache, "
          f"{len(todo)} can goi ({n_batches} lo x{batch_size})")

    t0 = time.time()
    done = 0
    for bi, start in enumerate(range(0, len(todo), batch_size), 1):
        batch_ids = todo[start:start + batch_size]
        narratives = [topics[tid] for tid in batch_ids]

        try:
            out, meta = gemini.chat_json(
                model, schema.BATCH_SYSTEM_PROMPT,
                schema.batch_user_prompt(narratives),
                schema.batch_schema(len(batch_ids)))
        except gemini.GeminiError as e:
            # Loi ha tang (het quota, mang...) la TAM THOI — KHONG ghi vao
            # recs, de lan chay sau (khong can --force) tu dong thu lai thay
            # vi bi coi la "da xong" mai mai. Khac voi nhanh hong ben duoi
            # (JSON sai dinh dang / khop index that bai), von it co co hoi
            # tu sua khi chay lai voi cung dau vao nen van duoc cache.
            done += len(batch_ids)
            print(f"  lo {bi}/{n_batches} LOI TAM THOI (se thu lai o lan chay "
                  f"sau): {e}", file=sys.stderr)
            continue

        per_topic_seconds = round(meta["seconds"] / len(batch_ids), 2)
        items = (out or {}).get("profiles") or []
        by_index: dict[int, dict] = {}
        for it in items:
            idx = it.get("index") if isinstance(it, dict) else None
            if isinstance(idx, int) and 0 <= idx < len(batch_ids) and idx not in by_index:
                by_index[idx] = it
            # index thieu / trung / ngoai khoang: bo qua — benh an tuong ung
            # se roi vao nhanh "hong" ben duoi thay vi khop nham benh nhan.

        for i, tid in enumerate(batch_ids):
            narrative = narratives[i]
            it = by_index.get(i)
            prof = {k: v for k, v in it.items() if k != "index"} if it else None

            rec: dict = {"seconds": per_topic_seconds,
                        "prompt_tokens": meta["prompt_tokens"],
                        "output_tokens": meta["output_tokens"],
                        "batch_size": len(batch_ids),
                        "profile": prof}
            if prof is not None and verify.schema_ok(prof):
                rec["clean"], rec["dropped"] = verify.verify_profile(prof, narrative)
            else:
                # Giu lai chuoi tho de go loi: biet model tra ve CAI GI khi
                # hong quan trong hon la chi biet rang no hong.
                rec["clean"], rec["dropped"] = None, None
                rec["raw"] = meta["raw"][:2000]
            recs[tid] = rec

        done += len(batch_ids)
        el = time.time() - t0
        print(f"  lo {bi}/{n_batches}  {done}/{len(todo)} benh an  {el:.0f}s  "
              f"({el/done:.1f}s/benh an)", flush=True)
        _save(path, ph, model, recs)

    # Goi lai vo dieu kien: neu lo cuoi cung loi tam thoi (khong _save() rieng),
    # file van phai phan anh dung recs hien tai truoc khi in thong bao ben duoi.
    _save(path, ph, model, recs)

    n_bad = sum(1 for r in recs.values() if r.get("clean") is None)
    n_drop = sum(len(r["dropped"]) for r in recs.values() if r.get("dropped"))
    print(f"Da ghi {path}")
    print(f"  {len(recs)} ban ghi, {n_bad} hong schema, {n_drop} gia tri bi vut "
          f"vi khong trich dan duoc")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=gemini.MODEL)
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--limit", type=int, default=None, help="chi chay N benh an dau")
    ap.add_argument("--profile-dir", default=PROFILE_DIR)
    ap.add_argument("--force", action="store_true", help="bo qua cache")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="so benh an moi lan goi Gemini (mac dinh 5)")
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! Dang chay tren TAP TEST 2022 — Phase 4 chi lam tren dev.",
              file=sys.stderr)

    if not gemini.KEYS:
        print("Khong co GEMINI_API_KEY_1/2/3 nao trong .env. Xem .env.example.",
              file=sys.stderr)
        return 1

    return run(args.model, args.year, args.limit, args.profile_dir, args.force,
              args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
