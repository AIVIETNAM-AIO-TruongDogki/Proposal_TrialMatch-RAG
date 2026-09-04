"""Phase 5 step 1 — four encoders, four RECIPES, one shared chunking scheme.

    python -m src.dense.encode --self-test          # run this FIRST
    python -m src.dense.encode --model bge-m3 --input data/jsonl/base --out ...

One recipe per model, per its own model card:
`bge-m3`/`qwen3` go through sentence-transformers' encode_query/encode_document,
which read the model's own config and apply the right prompt/prefix automatically.

`medcpt` does NOT use sentence-transformers: its card mandates raw CLS pooling
(`last_hidden_state[:,0,:]`), no normalization. Calling it through ST would
silently mean-pool and normalize instead — wrong on both counts, no error,
just a low score. It's also TWO separate models (query encoder, article
encoder); `_MedCPT.encode_query/document` call different models, and
`--self-test` verifies they actually are different.

`gemini` goes through the API (no GPU) and has its own asymmetry: task_type
must be RETRIEVAL_QUERY for narratives, RETRIEVAL_DOCUMENT for trials. It's a
4th comparison point, not a replacement for the other three — specs/05 asks
whether biomedical models beat general-purpose ones, and dropping BGE-M3/MedCPT
would erase that question.

Deliberate deviation from MedCPT's card: it uses max_length=64 for queries
(PubMed queries are short); ours are ~200 tokens, so 512 (the model's
positional limit) is used instead. Logged in docs/decisions/phase5-dense.md.

Chunking is by WORD, not by token: 320 words, 40 overlap, score = MAX across
chunks. Four encoders have four different tokenizers (XLM-R/BERT/Qwen/Gemini)
— chunking by each model's own tokens would create four different chunk sets,
comparing four different datasets. Word boundaries are identical by
definition. 320 words ~ 420 clinical tokens, safely under MedCPT's 512 cap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

CHUNK_WORDS = 320
CHUNK_OVERLAP = 40
BATCH = 32

MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "medcpt": "ncbi/MedCPT-Query-Encoder|ncbi/MedCPT-Article-Encoder",
    "gemini": "gemini-embedding-001",
}

# API embedding works but isn't fast enough for this corpus. Measured limits:
#   * max 100 texts/call (250 -> 400 INVALID_ARGUMENT)
#   * quota EmbedContentRequestsPerMinutePerUserPerProjectPerModel = 100/min
# Trap: the quota counts per TEXT, not per call — confirmed by the API's own
# 400 error text ("at most 100 requests") and empirically (6 concurrent
# 100-text calls -> 5 immediate 429s). Real throughput is 100 texts/minute:
#     subsample  49,652 chunks -> 8.3h;  full corpus ~400,000 chunks -> 67h
# vs. GPU: 16min and ~2h — 30x slower. So `gemini` is excluded from
# bench.py's default model list (code kept — viable once on a paid tier, via
# `--models gemini`). Fine for encoding QUERIES (75 of them); only encoding
# the CORPUS is the bottleneck.
GEMINI_EMBED_BATCH = 100
GEMINI_CONCURRENCY = 4


def chunk_words(text: str, size: int = CHUNK_WORDS,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split by word. Always returns at least one chunk (possibly empty), so every doc gets a vector."""
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)]
    step = size - overlap
    return [" ".join(w[i:i + size]) for i in range(0, len(w), step)
            if w[i:i + size]]


class _ST:
    """bge-m3 / Qwen3-Embedding — let sentence-transformers apply the recipe."""

    BATCH = BATCH   # GPU memory limit

    def __init__(self, name: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=device,
                                     model_kwargs={"dtype": "float16"})
        self.name = name
        # Qwen3-Embedding is a 0.6B DECODER, far more memory-hungry than
        # bge-m3's 568M encoder: at BATCH=32 it OOMs on a 7.62GB GPU.
        # Overridden at the INSTANCE level since both models share this class.
        #
        # Safe for comparability: each text is encoded INDEPENDENTLY, no
        # cross-attention within a batch — changing batch size changes speed,
        # not the vector. The speed difference still shows honestly in s/doc.
        self.BATCH = 8 if "Qwen3" in name else BATCH

    # bs=None -> use self.BATCH. Not defaulted to the BATCH constant here: a
    # default is fixed at class DEFINITION time, so it would bypass qwen3's
    # instance override and OOM on any call that omits bs (e.g. search.encode_query).
    def encode_query(self, texts: list[str], bs: int | None = None) -> np.ndarray:
        return self.m.encode_query(texts, batch_size=bs or self.BATCH,
                                   convert_to_numpy=True,
                                   normalize_embeddings=True,
                                   show_progress_bar=False)

    def encode_document(self, texts: list[str], bs: int | None = None) -> np.ndarray:
        return self.m.encode_document(texts, batch_size=bs or self.BATCH,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)


class _MedCPT:
    """MedCPT — two models, raw CLS, no normalization. See module docstring."""

    MAX_LEN = 512  # deliberate deviation from the card (64)
    BATCH = BATCH  # GPU memory limit

    def __init__(self, pair: str, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        q_name, d_name = pair.split("|")
        assert q_name != d_name, "MedCPT phai la HAI model khac nhau"
        self.torch = torch
        self.device = device
        self.q_tok = AutoTokenizer.from_pretrained(q_name)
        self.q_mod = AutoModel.from_pretrained(q_name).to(device).eval().half()
        self.d_tok = AutoTokenizer.from_pretrained(d_name)
        self.d_mod = AutoModel.from_pretrained(d_name).to(device).eval().half()
        self.name = pair

    def _encode(self, texts, tok, mod, bs):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, padding=True,
                      max_length=self.MAX_LEN, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                h = mod(**enc).last_hidden_state[:, 0, :]  # raw CLS, no normalization
            out.append(h.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)

    def encode_query(self, texts: list[str], bs: int = BATCH) -> np.ndarray:
        return self._encode(texts, self.q_tok, self.q_mod, bs)

    def encode_document(self, texts: list[str], bs: int = BATCH) -> np.ndarray:
        return self._encode(texts, self.d_tok, self.d_mod, bs)


class _GeminiEmbed:
    """gemini-embedding-001 via API — no GPU.

    Asymmetric like MedCPT, and just as silently wrong if mishandled:
    task_type must be RETRIEVAL_QUERY for narratives, RETRIEVAL_DOCUMENT for
    trials. Using the wrong one for one side raises no error, just a lower
    score — exactly the class of bug --self-test exists to catch.

    Key rotation and retryDelay reuse src/extraction/gemini.py's mechanism
    rather than reimplementing it.
    """

    # 100 = the API's max allowed. For an API, batch size is a REQUEST-count
    # question, not GPU memory — using 32 like the other encoders would
    # triple the request count for nothing.
    BATCH = GEMINI_EMBED_BATCH

    def __init__(self, name: str, device: str = "cuda"):
        from google import genai
        from src.extraction import gemini as gem
        if not gem.KEYS:
            raise RuntimeError("Khong co GEMINI_API_KEY_* trong .env")
        self.genai = genai
        self.gem = gem
        self.name = name

    def _one(self, chunk: list[str], task: str) -> list:
        from google.genai import errors, types
        for _ in range(len(self.gem.KEYS)):
            key = next(self.gem._cycle)
            client = self.genai.Client(api_key=key)   # kept alive
            try:
                r = client.models.embed_content(
                    model=self.name, contents=chunk,
                    config=types.EmbedContentConfig(task_type=task))
                return [e.values for e in r.embeddings]
            except errors.APIError as e:
                if e.code in self.gem._RETRYABLE_CODES:
                    time.sleep(self.gem._retry_delay_seconds(e))
                    continue
                raise
        raise RuntimeError(f"embed: het {len(self.gem.KEYS)} key deu loi")

    def _embed(self, texts: list[str], task: str, bs: int = GEMINI_EMBED_BATCH):
        """Calls IN PARALLEL — measured: one 100-text request takes ~23s.

        Quota is 100 requests/MINUTE, but sequential calls only reach
        ~2.6/minute because latency eats the time. The real constraint is
        LATENCY, not quota — concurrency is required to actually use the
        quota granted. Result order is kept by numbering pieces, not by
        thread completion order.
        """
        from concurrent.futures import ThreadPoolExecutor

        pieces = [[t or " " for t in texts[i:i + bs]]      # API rejects empty strings
                  for i in range(0, len(texts), bs)]
        if not pieces:
            return np.zeros((0, 3072), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=GEMINI_CONCURRENCY) as ex:
            results = list(ex.map(lambda p: self._one(p, task), pieces))
        out = [v for r in results for v in r]
        return np.asarray(out, dtype=np.float32)

    # bs is deliberately IGNORED: for an API, batch size is a request-count
    # question, not GPU memory, so it always uses the max allowed (100).
    # Accepted as a parameter only to keep the same call signature as the
    # other encoders.
    def encode_query(self, texts: list[str], bs: int = GEMINI_EMBED_BATCH) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")

    def encode_document(self, texts: list[str], bs: int = GEMINI_EMBED_BATCH) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")


def load_encoder(key: str, device: str = "cuda"):
    name = MODELS[key]
    if key == "medcpt":
        return _MedCPT(name, device)
    if key == "gemini":
        return _GeminiEmbed(name, device)
    return _ST(name, device)


# --- Sanity check before encoding anything -----------------------------------

_PARA_A = "The patient is a 45-year-old man with anaplastic astrocytoma of the spine."
_PARA_B = "A 45-year-old male presenting with spinal anaplastic astrocytoma."
_UNREL = "Randomized trial of dietary sodium reduction in healthy adolescents."


def self_test(device: str = "cuda") -> bool:
    """Encode a paraphrase pair + an unrelated pair, expect sim(para) > sim(unrel).

    A pooling bug raises no exception — it just makes the model look bad.
    Two seconds here is cheaper than an hour of encoding gone wrong.
    """
    ok_all = True
    for key in MODELS:
        try:
            enc = load_encoder(key, device)
            q = enc.encode_query([_PARA_A])
            d = enc.encode_document([_PARA_B, _UNREL])
            qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
            s_para, s_unrel = float(qn[0] @ dn[0]), float(qn[0] @ dn[1])
            ok = s_para > s_unrel
            extra = ""
            if key == "medcpt":
                extra = f"  (2 model rieng: {enc.q_mod is not enc.d_mod})"
            print(f"  {key:8s} dim={d.shape[1]:4d}  sim(para)={s_para:+.4f}  "
                  f"sim(unrel)={s_unrel:+.4f}  {'DAT' if ok else 'KHONG DAT'}{extra}")
            ok_all &= ok
            del enc
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {key:8s} LOI: {e}", file=sys.stderr)
            ok_all = False
    return ok_all


# --- Encode one JSONL directory ----------------------------------------------

def read_jsonl_dir(d: str, keep: set[str] | None = None,
                   shard: int | None = None) -> tuple[list[str], list[str]]:
    """shard=N reads only docs{N:02d}.jsonl. See `shard_path`/`merge_shards`."""
    ids, texts = [], []
    files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
    if shard is not None:
        want = f"docs{shard:02d}.jsonl"
        if want not in files:
            raise SystemExit(f"Khong co {d}/{want} (co: {', '.join(files)})")
        files = [want]
    for f in files:
        if not f.endswith(".jsonl"):
            continue
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                if keep is not None and o["id"] not in keep:
                    continue
                ids.append(o["id"])
                texts.append(o.get("contents") or "")
    return ids, texts


def encode_docs(enc, ids: list[str], texts: list[str], bs: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per-CHUNK vectors, each chunk's owning doc index).

    Kept at chunk granularity rather than pre-pooled: the final score is MAX
    across chunks, computed after the query dot-product, not before.
    """
    bs = bs or getattr(enc, "BATCH", BATCH)
    chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in chunk_words(t):
            chunks.append(c)
            owner.append(i)

    vecs, t0 = [], time.time()
    for i in range(0, len(chunks), bs):
        vecs.append(enc.encode_document(chunks[i:i + bs], bs))
        if i and i % (bs * 200) == 0:
            done = i + bs
            el = time.time() - t0
            print(f"    {done:,}/{len(chunks):,} chunk  {el:.0f}s  "
                  f"({done/el:.0f} chunk/s)", flush=True)
    V = np.concatenate(vecs).astype(np.float16)
    print(f"  {len(ids):,} doc -> {len(chunks):,} chunk, {V.shape[1]}d, "
          f"{time.time()-t0:.0f}s ({len(chunks)/(time.time()-t0):.0f} chunk/s)")
    return V, np.asarray(owner, dtype=np.int32)


# --- Sharded encoding, with checkpointing ------------------------------------
#
# WHY: encoding the full corpus takes ~3.5h and `encode_docs` only writes to
# disk ONCE at the end — one crash loses the whole run. It also keeps every
# fp32 vector in RAM, and np.concatenate makes a second copy: 406k chunks x
# 1024 x 4 bytes = 1.66GB x2, plus the fp16 copy, peaking ~4.8GB on an 8GB
# machine.
#
# Sharding fixes both: peak RAM drops to ~1GB, and each shard saves as soon
# as it's done. `data/jsonl/*` is already split into 4 shards of 100,000 docs.

def shard_path(out: str, shard: int) -> str:
    base = out[:-4] if out.endswith(".npz") else out
    return f"{base}.shard{shard:02d}.npz"


def merge_shards(out: str, model: str, keep_shards: bool = False) -> int:
    """Merge shard files into one .npz.

    THE ONE EASY BUG: `owner` is a LOCAL doc index within each shard.
    Concatenated directly, every chunk in shards 1..3 would point back to
    shard 0's documents — no exception, just a plausible-looking wrong score.
    Each shard's doc count must be offset first.
    """
    base = out[:-4] if out.endswith(".npz") else out
    paths = sorted(glob.glob(f"{base}.shard*.npz"))
    if not paths:
        raise SystemExit(f"Khong tim thay shard nao khop {base}.shard*.npz")

    ids_all: list[str] = []
    vecs, owners = [], []
    off = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        sid = list(z["ids"])
        vecs.append(z["vecs"])
        owners.append(z["owner"].astype(np.int64) + off)   # <- offset, never raw
        ids_all.extend(sid)
        off += len(sid)
        print(f"  {os.path.basename(p):40s} {len(sid):7,} doc  {z['vecs'].shape[0]:7,} chunk"
              f"  offset -> {off:,}")

    V = np.concatenate(vecs)
    owner = np.concatenate(owners).astype(np.int32)
    assert owner.max() == len(ids_all) - 1, (
        f"owner.max()={owner.max()} nhung co {len(ids_all)} doc — offset sai")
    assert len(owner) == V.shape[0], "so owner khong khop so chunk"
    save(out, ids_all, V, owner, model)
    if not keep_shards:
        for p in paths:
            os.remove(p)
        print(f"  da xoa {len(paths)} file shard")
    return 0


def save(out: str, ids: list[str], V: np.ndarray, owner: np.ndarray, model: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, vecs=V, owner=owner, ids=np.array(ids, dtype=object),
             model=model, chunk_words=CHUNK_WORDS, chunk_overlap=CHUNK_OVERLAP)
    mb = os.path.getsize(out + (".npz" if not out.endswith(".npz") else "")) / 1e6
    print(f"Da ghi {out}  ({mb:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--input", help="thu muc JSONL, vd data/jsonl/base")
    ap.add_argument("--out", help="duong dan .npz")
    ap.add_argument("--keep", default=None,
                    help="file .txt moi dong mot nct_id — chi ma hoa cac id nay")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--shard", type=int, default=None,
                    help="chi ma hoa docs{N:02d}.jsonl, ghi <out>.shardNN.npz")
    ap.add_argument("--merge", action="store_true",
                    help="gop <out>.shard*.npz thanh <out>")
    ap.add_argument("--keep-shards", action="store_true",
                    help="giu lai file shard sau khi gop")
    args = ap.parse_args()

    if args.merge:
        if not (args.model and args.out):
            ap.error("--merge can --model va --out")
        return merge_shards(args.out, MODELS[args.model], args.keep_shards)

    if args.self_test:
        print(f"Self-test {len(MODELS)} encoder (paraphrase phai gan hon unrelated):")
        ok = self_test(args.device)
        print("TAT CA DAT" if ok else "CO ENCODER KHONG DAT — dung lai truoc khi ma hoa")
        return 0 if ok else 1

    if not (args.model and args.input and args.out):
        ap.error("can --model, --input, --out (hoac --self-test)")

    keep = None
    if args.keep:
        keep = {l.strip() for l in open(args.keep, encoding="utf-8") if l.strip()}
        print(f"Loc theo {args.keep}: {len(keep):,} id")

    out = shard_path(args.out, args.shard) if args.shard is not None else args.out
    if args.shard is not None and os.path.exists(out):
        print(f"{out} da co — bo qua (xoa file neu muon ma hoa lai)")
        return 0

    ids, texts = read_jsonl_dir(args.input, keep, args.shard)
    where = f"{args.input}" + (f" shard {args.shard}" if args.shard is not None else "")
    print(f"{args.model}: {len(ids):,} doc tu {where}")
    enc = load_encoder(args.model, args.device)
    V, owner = encode_docs(enc, ids, texts, args.batch)
    save(out, ids, V, owner, MODELS[args.model])
    return 0


if __name__ == "__main__":
    sys.exit(main())
