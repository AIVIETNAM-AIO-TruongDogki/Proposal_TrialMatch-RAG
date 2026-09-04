"""Phase 5 step 2 — exact dense search via matmul, no FAISS.

    python -m src.dense.search --vecs indexes/dense/bge-m3.base.npz \
        --model bge-m3 --out runs/dense.dev.txt

Why not FAISS/HNSW: 375,580 x 1024-dim x fp16 = 769 MB, fits an 8GB GPU with
room to spare. 75 queries against the whole corpus plus torch.topk takes
seconds. Brute force is exact; approximate-index error would blend into the
model-vs-model gap on the ablation ladder — rung 2 vs. rung 1 differences
would be from the MODEL or the INDEX, indistinguishably. Qdrant/FAISS is
Phase 10's deployment question, not Phase 5's research question.

Score = MAX across chunks: a trial is split into chunks, and its score is the
HIGHEST-scoring chunk, taken AFTER the query dot-product, not by pooling
vectors first — pooling first blurs exactly which passage matched.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src.dense import encode
from src.eval import data, run_io


def load_vecs(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=True)
    return z["vecs"], z["owner"], list(z["ids"])


def search(vec_path: str, model_key: str, topics: dict[str, str],
           depth: int = 1000, device: str = "cuda") -> dict[str, dict[str, float]]:
    import torch

    V, owner, ids = load_vecs(vec_path)
    print(f"  {len(ids):,} doc / {V.shape[0]:,} chunk / {V.shape[1]}d")

    enc = encode.load_encoder(model_key, device)
    qids = sorted(topics)
    Q = enc.encode_query([topics[q] for q in qids])

    # Free GPU memory BEFORE allocating Vt. The encoder (1.1GB fp16 for
    # bge-m3) has nothing left to do after encoding 75 queries, but left
    # alive it stacks with Vt (~920MB for the full corpus) and the temp
    # float32 copy during normalization (~819MB/chunk) on the same 7.62GB.
    del enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Normalize BOTH sides here rather than trusting the encoder: MedCPT
    # deliberately doesn't normalize, so cosine must be made explicit here.
    Qt = torch.tensor(np.asarray(Q, dtype=np.float32), device=device)
    Qt = torch.nn.functional.normalize(Qt, dim=1).half()

    # Normalize in CHUNKS, keep fp16. Calling .float() on the whole matrix
    # allocates a float32 copy: 375,580 chunks x 3072 dims is 4.6GB, and doing
    # this inside the query loop would re-allocate it PER QUERY -> guaranteed
    # OOM on an 8GB GPU. Only shows up at full-corpus scale, not on a subsample.
    Vt = torch.tensor(V, device=device, dtype=torch.float16)
    for s in range(0, Vt.shape[0], 200_000):
        Vt[s:s + 200_000] = torch.nn.functional.normalize(
            Vt[s:s + 200_000].float(), dim=1).half()

    own = torch.tensor(owner.astype(np.int64), device=device)
    n_docs = len(ids)

    run: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for i, qid in enumerate(qids):
        sims = (Vt @ Qt[i]).float()                      # fp16 matmul, fp32 result
        best = torch.full((n_docs,), -1e9, device=device)
        best.scatter_reduce_(0, own, sims, reduce="amax")  # MAX across chunks
        k = min(depth, n_docs)
        sc, idx = torch.topk(best, k)
        run[qid] = {ids[j]: float(s) for j, s in zip(idx.tolist(), sc.tolist())}
    print(f"  {len(qids)} truy van, {time.time()-t0:.1f}s")
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--model", required=True, choices=list(encode.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.year == data.TEST_YEAR:
        print("!! Dang chay tren TAP TEST 2022 — chi cham MOT LAN o Phase 11.",
              file=sys.stderr)

    topics = data.load_topics(args.year)
    run = search(args.vecs, args.model, topics, args.depth, args.device)
    tag = args.tag or args.out.split("/")[-1].rsplit(".", 1)[0]
    run_io.write_run(args.out, run, tag, depth=args.depth)
    print(f"Da ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
