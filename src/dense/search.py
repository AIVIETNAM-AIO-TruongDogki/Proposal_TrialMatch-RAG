"""Phase 5 buoc 2 — tim kiem dense bang matmul CHINH XAC, khong FAISS.

    python -m src.dense.search --vecs indexes/dense/bge-m3.base.npz \
        --model bge-m3 --out runs/dense.dev.txt

VI SAO KHONG FAISS/HNSW
------------------------
375.580 x 1024 chieu x fp16 = 769 MB — vua GPU 8 GB con du cho. 75 truy van
nhan ca corpus roi torch.topk mat vai giay. Brute force la CHINH XAC.

FAISS/HNSW dua vao sai so xap xi, va sai so do se tron vao chenh lech
model-vs-model tren thang ablation: khong con biet bac 2 khac bac 1 vi MODEL
hay vi THAM SO INDEX. Bo mot phu thuoc va bo luon mot bien gay nhieu.
Qdrant/FAISS la cau hoi trien khai cua Phase 10, khong phai cau hoi nghien
cuu cua Phase 5.

DIEM = MAX QUA CAC CHUNK
-------------------------
Mot trial duoc cat thanh nhieu chunk; diem cua trial la diem CAO NHAT trong
cac chunk cua no. Phai max SAU khi nhan voi truy van, khong phai gop vector
truoc — gop truoc se lam nhoe dung cai doan van khop nhat.
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

    # Tra GPU lai TRUOC khi cap phat Vt. Encoder (1,1 GB fp16 voi bge-m3) khong
    # con viec gi sau khi ma hoa 75 truy van, nhung neu de no song thi no cong
    # don voi Vt (~920 MB o toan corpus) va ban sao float32 tam trong vong chuan
    # hoa (~819 MB moi mieng) tren cung 7,62 GB.
    del enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Chuan hoa CA HAI phia o day thay vi tin vao encoder: MedCPT co y khong
    # normalize, nen cosine phai duoc lam ro rang tai diem so sanh.
    Qt = torch.tensor(np.asarray(Q, dtype=np.float32), device=device)
    Qt = torch.nn.functional.normalize(Qt, dim=1).half()

    # Chuan hoa theo TUNG MIENG roi giu fp16. Goi .float() tren ca ma tran se
    # cap phat mot ban sao float32: voi 375.580 chunk x 3072 chieu la 4,6 GB,
    # va neu de trong vong lap thi cap phat lai cho MOI truy van -> OOM chac
    # chan tren GPU 8 GB. Loi nay chi lo ra o quy mo toan corpus, khong lo o
    # subsample, nen phai chan tu day.
    Vt = torch.tensor(V, device=device, dtype=torch.float16)
    for s in range(0, Vt.shape[0], 200_000):
        Vt[s:s + 200_000] = torch.nn.functional.normalize(
            Vt[s:s + 200_000].float(), dim=1).half()

    own = torch.tensor(owner.astype(np.int64), device=device)
    n_docs = len(ids)

    run: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for i, qid in enumerate(qids):
        sims = (Vt @ Qt[i]).float()                      # matmul fp16, ket qua fp32
        best = torch.full((n_docs,), -1e9, device=device)
        best.scatter_reduce_(0, own, sims, reduce="amax")  # MAX qua chunk
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
