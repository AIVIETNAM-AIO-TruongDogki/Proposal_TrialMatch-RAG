"""Phase 3 buoc 3 — dung index Lucene tu JSONL.

    python -m src.retrieval.build_index                    # ban base
    python -m src.retrieval.build_index --with-criteria    # ban de ablate

Boc lenh `pyserini.index.lucene`. Viet thanh module de tham so index duoc ghi
lai cung cho voi phan con lai cua pipeline, thay vi nam trong lich su shell.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def build(input_dir: str, index_dir: str, threads: int = 16) -> int:
    if not os.path.isdir(input_dir):
        print(f"Khong thay {input_dir}. Chay truoc:\n"
              f"  python -m src.retrieval.export_corpus", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        # storePositions/Docvectors can cho reranking va phan tich sau nay;
        # storeRaw de doc lai van ban da index khi go loi.
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    print(" ".join(cmd) + "\n", flush=True)

    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\nIndex that bai (ma loi {rc}).", file=sys.stderr)
        return rc

    size = sum(os.path.getsize(os.path.join(index_dir, f))
               for f in os.listdir(index_dir)
               if os.path.isfile(os.path.join(index_dir, f)))
    print(f"\nXong trong {time.time()-t0:.0f}s. Index: {index_dir} "
          f"({size/1e9:.2f} GB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-criteria", action="store_true")
    ap.add_argument("--input", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    variant = "crit" if args.with_criteria else "base"
    return build(args.input or f"data/jsonl/{variant}",
                 args.index or f"indexes/bm25-{variant}",
                 args.threads)


if __name__ == "__main__":
    sys.exit(main())
