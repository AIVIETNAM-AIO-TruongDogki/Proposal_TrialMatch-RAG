"""Phase 10 — index dense THUONG TRU cho demo web song.

`src/dense/search.py::search()` la mot batch script: no load lai toan bo npz
+ khoi tao lai encoder tu dau MOI LAN goi, roi CHU DONG giai phong encoder
truoc khi cap phat ma tran corpus (xem `del enc` o do). Hop ly cho 75 truy van
nghien cuu chay MOT LAN, khong dung duoc cho phuc vu web — moi request se mat
thoi gian load 816MB + khoi tao lai transformer.

`LiveDenseIndex` giu CA vector corpus (chuan hoa MOT LAN luc khoi dong) LAN
encoder song song thuong tru trong GPU/CPU suot vong doi tien trinh. Day la
thay doi HA TANG (batch-script -> service thuong tru), khong phai thay doi
THUAT TOAN: cung model, cung chunking 320/40, cung diem = max qua chunk, cung
matmul chinh xac khong FAISS da chon o Phase 5 (xem docstring cua search.py
ve ly do tu choi FAISS/HNSW — quyet dinh do khong bi mo lai o day).
"""

from __future__ import annotations

import numpy as np
import torch

from src.dense import encode
from src.dense.search import load_vecs


class LiveDenseIndex:
    def __init__(self, vec_path: str, model_key: str, device: str = "cuda"):
        V, owner, ids = load_vecs(vec_path)
        self.device = device
        self.ids = ids
        self.owner = torch.tensor(owner.astype(np.int64), device=device)
        self.n_docs = len(ids)
        self.enc = encode.load_encoder(model_key, device)

        # Chuan hoa MOT LAN o day va giu nguyen suot vong doi tien trinh — khac
        # search.py, noi moi lan goi la mot lan chay doc lap nen chuan hoa lai
        # khong ton kem gi (chi 75 truy van/lan). O day co the co hang tram
        # request, nen chuan hoa lai moi request la lang phi khong can thiet.
        Vt = torch.tensor(V, device=device, dtype=torch.float16)
        for s in range(0, Vt.shape[0], 200_000):
            Vt[s:s + 200_000] = torch.nn.functional.normalize(
                Vt[s:s + 200_000].float(), dim=1).half()
        self.Vt = Vt
        print(f"LiveDenseIndex: {self.n_docs:,} doc / {Vt.shape[0]:,} chunk / "
              f"{Vt.shape[1]}d thuong tru tren {device}")

    def query(self, text: str, depth: int = 1000) -> dict[str, float]:
        """Ma hoa MOT truy van moi va tra ve {nct_id: diem}, diem = max qua chunk.

        Khong load lai gi ca — chi ham nay chay tren duong request.
        """
        q = self.enc.encode_query([text])
        Qt = torch.tensor(np.asarray(q, dtype=np.float32), device=self.device)
        Qt = torch.nn.functional.normalize(Qt, dim=1).half()[0]

        sims = (self.Vt @ Qt).float()
        best = torch.full((self.n_docs,), -1e9, device=self.device)
        best.scatter_reduce_(0, self.owner, sims, reduce="amax")
        k = min(depth, self.n_docs)
        sc, idx = torch.topk(best, k)
        return {self.ids[j]: float(s) for j, s in zip(idx.tolist(), sc.tolist())}
