"""Phase 4 buoc 3 — chay trich xuat, co cache.

    python -m src.extraction.extract --year 2021 --model qwen2.5:7b-instruct

Ghi ra data/profiles/{year}.{model}.json. Moi ban ghi giu CA HAI phien ban:
`profile` tho tu model va `clean` sau khi loc grounding, kem `dropped` — danh
sach nhung gi bi vut vi khong trich dan duoc.

Giu lai phan bi vut la co y. "Ty le grounding 91%" chi la mot con so; danh sach
9% bi vut moi cho biet model bia KIEU GI, va do la thu can cho hand-audit o
buoc 5. Neu chi ghi ban da loc thi ta xoa mat bang chung cua chinh phep do.

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
from src.extraction import ollama, schema, verify

PROFILE_DIR = "data/profiles"


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


def run(model: str, year: int, limit: int | None = None,
        profile_dir: str = PROFILE_DIR, force: bool = False) -> int:
    topics = data.load_topics(year)
    if limit:
        topics = dict(list(topics.items())[:limit])

    ph = schema.prompt_hash()
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, f"{year}.{model.replace(':', '_')}.json")
    recs = {} if force else load_cache(path, ph)

    todo = [t for t in topics if t not in recs]
    print(f"{model}: {len(topics)} benh an, {len(recs)} da co cache, "
          f"{len(todo)} can goi")

    t0 = time.time()
    for i, tid in enumerate(todo, 1):
        narrative = topics[tid]
        prof, meta = ollama.chat_json(
            model, schema.SYSTEM_PROMPT,
            schema.USER_TEMPLATE.format(narrative=narrative),
            schema.PROFILE_SCHEMA)

        rec: dict = {"seconds": meta["seconds"],
                     "prompt_tokens": meta["prompt_tokens"],
                     "output_tokens": meta["output_tokens"],
                     "profile": prof}
        if prof is not None and verify.schema_ok(prof):
            rec["clean"], rec["dropped"] = verify.verify_profile(prof, narrative)
        else:
            # Giu lai chuoi tho de go loi: biet model tra ve CAI GI khi hong
            # quan trong hon la chi biet rang no hong.
            rec["clean"], rec["dropped"] = None, None
            rec["raw"] = meta["raw"][:2000]

        recs[tid] = rec
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el:.0f}s  ({el/i:.1f}s/benh an)", flush=True)

        json.dump({"prompt_hash": ph, "model": model, "records": recs},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    n_bad = sum(1 for r in recs.values() if r.get("clean") is None)
    n_drop = sum(len(r["dropped"]) for r in recs.values() if r.get("dropped"))
    print(f"Da ghi {path}")
    print(f"  {len(recs)} ban ghi, {n_bad} hong schema, {n_drop} gia tri bi vut "
          f"vi khong trich dan duoc")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--limit", type=int, default=None, help="chi chay N benh an dau")
    ap.add_argument("--profile-dir", default=PROFILE_DIR)
    ap.add_argument("--force", action="store_true", help="bo qua cache")
    ap.add_argument("--unload", action="store_true",
                    help="nha model khoi VRAM khi xong (truoc khi chay encoder)")
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! Dang chay tren TAP TEST 2022 — Phase 4 chi lam tren dev.",
              file=sys.stderr)

    if not ollama.is_up():
        print("Ollama khong chay. Thu: ollama serve", file=sys.stderr)
        return 1
    have = ollama.list_models()
    if args.model not in have:
        print(f"Chua co model {args.model}. Dang co:\n  " + "\n  ".join(have),
              file=sys.stderr)
        return 1

    rc = run(args.model, args.year, args.limit, args.profile_dir, args.force)
    if args.unload:
        ollama.unload(args.model)
        print(f"Da nha {args.model} khoi VRAM")
    return rc


if __name__ == "__main__":
    sys.exit(main())
