"""Phase 10 — a PERSISTENT dense index for the live web demo.

`src/dense/search.py::search()` is a batch script: it reloads the whole npz
and reinitializes the encoder EVERY call, then frees the encoder before
allocating the corpus matrix (see `del enc` there). Fine for 75 research
queries run once; unusable for serving the web — every request would pay to
load 816MB and reinit a transformer.

`LiveDenseIndex` keeps both the corpus vectors (normalized ONCE at startup)
and the encoder resident in GPU/CPU memory for the process lifetime. This is
an INFRASTRUCTURE change (batch script -> long-lived service), not an
ALGORITHM change: same model, same 320/40 chunking, same max-across-chunks
score, same exact matmul search chosen in Phase 5 (see search.py's docstring
for why FAISS/HNSW was rejected — that decision isn't reopened here).
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

        # Normalized ONCE here and kept for the process lifetime — unlike
        # search.py, where each run is independent so renormalizing costs
        # nothing (only 75 queries/run). Here there can be hundreds of
        # requests, so renormalizing per request would be needless waste.
        Vt = torch.tensor(V, device=device, dtype=torch.float16)
        for s in range(0, Vt.shape[0], 200_000):
            Vt[s:s + 200_000] = torch.nn.functional.normalize(
                Vt[s:s + 200_000].float(), dim=1).half()
        self.Vt = Vt
        print(f"LiveDenseIndex: {self.n_docs:,} doc / {Vt.shape[0]:,} chunk / "
              f"{Vt.shape[1]}d thuong tru tren {device}")

    def query(self, text: str, depth: int = 1000) -> dict[str, float]:
        """Encode ONE new query, return {nct_id: score}, score = max across chunks.

        Nothing gets reloaded — this is the only function on the request path.
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
