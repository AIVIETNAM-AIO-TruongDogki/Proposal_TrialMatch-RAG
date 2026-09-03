"""Phase 5 buoc 1 — bon encoder, bon cong thuc RIENG, cung mot cach cat chunk.

    python -m src.dense.encode --self-test          # BAT BUOC chay truoc
    python -m src.dense.encode --model bge-m3 --input data/jsonl/base --out ...

MOI MODEL MOT CONG THUC — THEO CARD CUA CHINH TAC GIA
-----------------------------------------------------
`bge-m3` va `qwen3` di qua sentence-transformers `encode_query/encode_document`:
hai ham do doc `config_sentence_transformers.json` cua chinh model va tu ap
prompt/prefix dung cach. Tu truyen prompt_name bang tay la doan thay tac gia.

`medcpt` KHONG dung sentence-transformers: card cua no quy dinh pooling CLS
tho (`last_hidden_state[:,0,:]`), KHONG normalize. Goi qua ST se lam mean
pooling va normalize -> sai ca hai, khong nem loi nao, chi ra diem thap.

`gemini` di qua API (khong dung GPU), va cung BAT DOI XUNG: task_type phai la
RETRIEVAL_QUERY cho benh an, RETRIEVAL_DOCUMENT cho trial. Day la ung vien
THU TU, cong them vao phep so sanh chu khong thay the ba encoder kia — cau hoi
cua specs/05 la "model y sinh co thang model tong quat khong", va bo BGE-M3
voi MedCPT di thi khong con cau hoi do nua.

MEDCPT BAT DOI XUNG — KHANG DINH BANG CODE, KHONG DUA VAO CAN THAN
------------------------------------------------------------------
MedCPT la HAI model: Query-Encoder cho benh an, Article-Encoder cho trial.
Dung nham mot encoder cho ca hai phia se lam diem tut va dan thang toi ket
luan "model y sinh thua model tong quat" — dung dieu Phase 5 sinh ra de KIEM
CHUNG chu khong phai de gia dinh. Vi vay `_MedCPT.encode_query/document` goi
hai model khac nhau va `--self-test` kiem tra chung khong phai mot.

LECH CO Y SO VOI CARD: card MedCPT dung max_length=64 cho query vi truy van
PubMed ngan. Benh an cua ta ~200 token, nen dung 512 (gioi han positional cua
model). Ghi trong docs/decisions/phase5-dense.md.

CAT CHUNK THEO TU, KHONG THEO TOKEN
------------------------------------
320 tu, chong lan 40, diem = MAX qua cac chunk. Bon encoder co bon tokenizer
khac nhau (XLM-R / BERT / Qwen / Gemini); cat theo token cua tung model se tao
ra bon tap chunk khac nhau va phep so sanh chay tren bon bo du lieu khac nhau.
Ranh gioi tu thi giong het nhau THEO DINH NGHIA. 320 tu ~ 420 token van ban
lam sang, an toan duoi tran 512 cua MedCPT.
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

# API EMBEDDING: DUNG DUOC NHUNG KHONG DU NHANH CHO CORPUS NAY
# -------------------------------------------------------------
# Gioi han do thuc te, khong doan:
#   * toi da 100 van ban moi lan goi (250 -> 400 INVALID_ARGUMENT)
#   * quota EmbedContentRequestsPerMinutePerUserPerProjectPerModel = 100/phut
#
# CAI BAY: quota do dem theo TUNG VAN BAN, khong phai theo lan goi. Chinh loi
# 400 cua API noi ro dieu do — "BatchEmbedContentsRequest.requests: at most 100
# requests" — no goi moi van ban la mot "request". Da xac nhan bang thuc nghiem:
# 6 lan goi dong thoi (moi lan 100 van ban) thi 5 dinh 429 ngay lap tuc.
#
# Vi vay thong luong that la 100 VAN BAN/phut:
#     subsample  49.652 chunk  ->  8,3 gio
#     toan corpus ~400.000 chunk -> 67 gio
# so voi GPU: 16 phut va ~2 gio. Cham hon 30 lan.
#
# Ket luan: `gemini` KHONG nam trong danh sach mac dinh cua bench.py. Code duoc
# giu lai vi no dung va se kha thi ngay khi len goi tra phi — chay bang
#     --models gemini
# Dung cho ma hoa TRUY VAN (75 cai) thi hoan toan on; chi ma hoa CORPUS moi tac.
GEMINI_EMBED_BATCH = 100
GEMINI_CONCURRENCY = 4


def chunk_words(text: str, size: int = CHUNK_WORDS,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cat theo tu. Tra ve it nhat mot chunk (co the rong) de moi doc luon co vector."""
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)]
    step = size - overlap
    return [" ".join(w[i:i + size]) for i in range(0, len(w), step)
            if w[i:i + size]]


class _ST:
    """bge-m3 / Qwen3-Embedding — de sentence-transformers tu ap cong thuc."""

    BATCH = BATCH   # gioi han bo nho GPU

    def __init__(self, name: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=device,
                                     model_kwargs={"dtype": "float16"})
        self.name = name
        # Qwen3-Embedding la DECODER 0,6B, ton bo nho hon nhieu so voi encoder
        # 568M cua bge-m3: o BATCH=32 no OOM tren GPU 7,62 GB. Ghi de o muc
        # INSTANCE vi hai model dung chung lop nay.
        #
        # An toan cho tinh so sanh: encoder xu ly tung van ban DOC LAP, khong co
        # attention cheo giua cac van ban trong batch — doi batch chi doi toc do,
        # khong doi vector. Chenh lech toc do van hien trung thuc o cot s/doc.
        self.BATCH = 8 if "Qwen3" in name else BATCH

    # bs=None -> lay self.BATCH. Khong dat mac dinh la hang BATCH o day: gia tri
    # mac dinh duoc chot luc DINH NGHIA lop, nen no se bo qua ghi de instance
    # cua qwen3 va OOM o duong goi khong truyen bs (vi du search.encode_query).
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
    """MedCPT — hai model, CLS tho, khong normalize. Xem docstring module."""

    MAX_LEN = 512  # lech co y so voi card (64)
    BATCH = BATCH  # gioi han bo nho GPU

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
                h = mod(**enc).last_hidden_state[:, 0, :]  # CLS tho, khong normalize
            out.append(h.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)

    def encode_query(self, texts: list[str], bs: int = BATCH) -> np.ndarray:
        return self._encode(texts, self.q_tok, self.q_mod, bs)

    def encode_document(self, texts: list[str], bs: int = BATCH) -> np.ndarray:
        return self._encode(texts, self.d_tok, self.d_mod, bs)


class _GeminiEmbed:
    """gemini-embedding-001 qua API — khong dung GPU.

    BAT DOI XUNG GIONG MEDCPT, va cung de sai am tham y het:
    `task_type` phai la RETRIEVAL_QUERY cho benh an va RETRIEVAL_DOCUMENT cho
    trial. Dung nham mot task_type cho ca hai phia khong nem loi nao — no chi
    lam diem tut, dung kieu loi ma cong --self-test sinh ra de bat.

    Xoay vong key va ton trong retryDelay dung theo co che cua
    src/extraction/gemini.py, khong viet lai.
    """

    # 100 = muc TOI DA API cho phep. Voi API, batch la chuyen so REQUEST chu
    # khong phai bo nho GPU, nen dung 32 nhu hai encoder kia se ton gap 3 lan
    # so request ma khong duoc gi.
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
            client = self.genai.Client(api_key=key)   # giu tham chieu song
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
        """Goi SONG SONG — do thuc te: mot request 100 van ban mat ~23 giay.

        Quota la 100 request/PHUT, nhung chay tuan tu chi dat ~2,6 request/phut
        vi do tre chiem het thoi gian. Rang buoc that la DO TRE, khong phai
        quota — nen phai goi dong thoi moi dung het phan quota da duoc cap.
        Giu thu tu ket qua bang cach danh so mieng, khong dua vao thu tu hoan
        thanh cua cac luong.
        """
        from concurrent.futures import ThreadPoolExecutor

        pieces = [[t or " " for t in texts[i:i + bs]]      # API tu choi chuoi rong
                  for i in range(0, len(texts), bs)]
        if not pieces:
            return np.zeros((0, 3072), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=GEMINI_CONCURRENCY) as ex:
            results = list(ex.map(lambda p: self._one(p, task), pieces))
        out = [v for r in results for v in r]
        return np.asarray(out, dtype=np.float32)

    # bs bi BO QUA co y: voi API, batch la chuyen so request chu khong phai
    # chuyen bo nho GPU, nen luon dung muc toi da 100 cho phep. Nhan tham so de
    # giu cung chu ky voi hai encoder kia.
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


# --- Cong kiem tra truoc khi ma hoa bat cu thu gi -----------------------------

_PARA_A = "The patient is a 45-year-old man with anaplastic astrocytoma of the spine."
_PARA_B = "A 45-year-old male presenting with spinal anaplastic astrocytoma."
_UNREL = "Randomized trial of dietary sodium reduction in healthy adolescents."


def self_test(device: str = "cuda") -> bool:
    """Ma hoa mot cap dien dat lai + mot cap khong lien quan, doi sim(para) > sim(unrel).

    Loi pooling KHONG nem ngoai le — no chi lam model do trong nhu kem. Hai
    giay o day re hon sau lan chay ma hoa hong.
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


# --- Ma hoa mot tap JSONL ----------------------------------------------------

def read_jsonl_dir(d: str, keep: set[str] | None = None,
                   shard: int | None = None) -> tuple[list[str], list[str]]:
    """shard=N chi doc docs{N:02d}.jsonl. Xem `shard_path`/`merge_shards`."""
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
    """Tra ve (vectors cua TUNG CHUNK, chi so doc cua tung chunk).

    Giu o muc chunk thay vi gop san: diem cuoi la MAX qua chunk, ma max phai
    tinh sau khi nhan voi truy van, khong phai truoc.
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


# --- Ma hoa theo shard, co checkpoint ----------------------------------------
#
# VI SAO CAN: ma hoa toan corpus mat ~3,5 gio va `encode_docs` chi ghi dia MOT
# LAN o cuoi — mot lan chet la mat ca lượt. Ngoai ra no giu toan bo vector fp32
# trong RAM roi `np.concatenate` tao ban sao thu hai: 406k chunk x 1024 x 4 byte
# = 1,66 GB x2, cong ban fp16 nua, dinh ~4,8 GB tren may chi con 8 GB kha dung.
#
# Chia theo shard giai quyet ca hai: dinh RAM xuong ~1 GB va moi shard duoc luu
# ngay khi xong. `data/jsonl/*` von da chia san 4 shard 100.000 doc.

def shard_path(out: str, shard: int) -> str:
    base = out[:-4] if out.endswith(".npz") else out
    return f"{base}.shard{shard:02d}.npz"


def merge_shards(out: str, model: str, keep_shards: bool = False) -> int:
    """Gop cac file shard thanh mot .npz duy nhat.

    CHO DE SAI DUY NHAT: `owner` la chi so tai lieu CUC BO trong tung shard. Noi
    thang se lam moi chunk cua shard 1..3 tro ve tai lieu cua shard 0 — khong nem
    ngoai le, chi lam diem so sai theo kieu van trong hop ly. Phai cong don so
    tai lieu cua cac shard truoc.
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
        owners.append(z["owner"].astype(np.int64) + off)   # <- offset, khong noi thang
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
