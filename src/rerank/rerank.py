"""Phase 7 — xep hang lai top-100 (bac 4 cua thang ablation).

    python -m src.rerank.rerank --self-test
    python -m src.rerank.rerank --run runs/hybrid.dev.txt --model medcpt \
        --out runs/hybrid_rerank.dev.txt
    python -m src.rerank.rerank --bench runs/hybrid.dev.txt   # ca ba model

BA HO MODEL, BA CACH GOI KHAC NHAU — DUNG GIA DINH MOT API HOP CA BA
---------------------------------------------------------------------
medcpt  MedCPT-Cross-Encoder: BERT cross-encoder, huan luyen CHUNG voi chinh
        retriever MedCPT. Diem = logit[0] tho.
bge     bge-reranker-v2-m3: cross-encoder tong quat. Diem = logit[0] tho.
qwen3   Qwen3-Reranker-0.6B: KHONG phai cross-encoder. No la mot LLM duoc hoi
        "co lien quan khong?" va diem = log P(yes) - log P(no) tren token ke
        tiep. Ap khuon cross-encoder len no se ra diem vo nghia ma khong nem
        loi nao.

TRAN CUNG — RECALL@100
-----------------------
Xep hang lai KHONG the cuu thu ma truy hoi khong bao gio tra ve. Recall@100
cua run dau vao la tran tuyet doi cua moi reranker o day. Bao cao no truoc
moi bang diem, neu khong se di tune mot thu da bi chan tran.

DO TRE LA KET QUA, KHONG PHAI CHU THICH
----------------------------------------
Mot reranker them 0.01 nDCG voi gia 40 lan do tre la mot ket qua AM cho mot
he thong dinh dung duoc that, va noi ro dieu do la mot dong gop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.corpus import store
from src.eval import data, metrics, run_io

DEPTH = 100
BATCH = 16

MODELS = {
    "medcpt": "ncbi/MedCPT-Cross-Encoder",
    "bge": "BAAI/bge-reranker-v2-m3",
    "qwen3": "Qwen/Qwen3-Reranker-0.6B",
}


class _CrossEncoder:
    """MedCPT / bge — cross-encoder that su, diem = logit[0]."""

    MAX_LEN = 512

    def __init__(self, name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.mod = (AutoModelForSequenceClassification
                    .from_pretrained(name).to(device).eval().half())

    def score(self, query: str, docs: list[str], bs: int = BATCH) -> list[float]:
        out = []
        for i in range(0, len(docs), bs):
            enc = self.tok([query] * len(docs[i:i + bs]), docs[i:i + bs],
                           truncation=True, padding=True, max_length=self.MAX_LEN,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.mod(**enc).logits
            out += logits[:, 0].float().cpu().tolist()
        return out


class _QwenReranker:
    """Qwen3-Reranker — LLM tra loi yes/no, diem = logP(yes) - logP(no).

    KHONG phai cross-encoder: khong co dau phan loai, chi co phan phoi token ke
    tiep. Dung AutoModelForSequenceClassification len no se hong am tham.
    """

    MAX_LEN = 2048
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the "
              "requirements based on the Query. Note that the answer can only be "
              "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.mod = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16).to(device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    def score(self, query: str, docs: list[str], bs: int = BATCH) -> list[float]:
        out = []
        for i in range(0, len(docs), bs):
            prompts = [f"{self.PREFIX}<Query>: {query}\n<Document>: {d}{self.SUFFIX}"
                       for d in docs[i:i + bs]]
            enc = self.tok(prompts, truncation=True, padding=True,
                           max_length=self.MAX_LEN, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                # logits_to_keep=1 — CHI tinh logits cho vi tri cuoi. Khong co no,
                # transformers tinh logits cho CA chuoi roi ta vut het tru vi tri
                # cuoi: 16 x 2048 x 151.669 x 2 byte = 9,94 GB thay vi 4,9 MB.
                # Day la nguyen nhan OOM that su, khong phai thieu VRAM.
                logits = self.mod(**enc, logits_to_keep=1).logits[:, -1, :].float()
            lp = self.torch.nn.functional.log_softmax(logits, dim=-1)
            out += (lp[:, self.yes] - lp[:, self.no]).cpu().tolist()
        return out


def load_reranker(key: str, device: str = "cuda"):
    name = MODELS[key]
    return (_QwenReranker(name, device) if key == "qwen3"
            else _CrossEncoder(name, device))


def self_test(device: str = "cuda") -> bool:
    """Cap lien quan phai duoc cham cao hon cap khong lien quan — cho ca ba ho."""
    q = "45-year-old man with anaplastic astrocytoma of the spine, prior radiation."
    rel = "Phase II study of temozolomide in adults with recurrent anaplastic astrocytoma."
    unrel = "Dietary sodium reduction in healthy adolescents: a randomized trial."
    ok_all = True
    for key in MODELS:
        try:
            rr = load_reranker(key, device)
            s_rel, s_unrel = rr.score(q, [rel, unrel])
            ok = s_rel > s_unrel
            print(f"  {key:8s} rel={s_rel:+8.3f}  unrel={s_unrel:+8.3f}  "
                  f"{'DAT' if ok else 'KHONG DAT'}")
            ok_all &= ok
            del rr
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {key:8s} LOI: {e}", file=sys.stderr)
            ok_all = False
    return ok_all


def doc_texts(nct_ids: list[str], conn) -> dict[str, str]:
    """Van ban trial de rerank — dung DUNG store.retrieval_text() nhu luc index."""
    out = {}
    for nct in nct_ids:
        t = store.get_trial(conn, nct)
        out[nct] = store.retrieval_text(t) if t else ""
    return out


def rerank_run(run: dict, topics: dict[str, str], key: str, depth: int = DEPTH,
               device: str = "cuda", db: str = "data/trials.db"
               ) -> tuple[dict, float]:
    conn = store.open_db(db)
    rr = load_reranker(key, device)
    out: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for n, tid in enumerate(sorted(run), 1):
        cand = [d for d, _ in sorted(run[tid].items(),
                                     key=lambda kv: (-kv[1], kv[0]))[:depth]]
        texts = doc_texts(cand, conn)
        scores = rr.score(topics[tid], [texts[d] for d in cand])
        out[tid] = dict(zip(cand, scores))
        if n % 25 == 0:
            print(f"    {n}/{len(run)} topic  {time.time()-t0:.0f}s", flush=True)
    el = time.time() - t0
    print(f"  {key}: {len(run)} topic x {depth} = {len(run)*depth:,} cap, "
          f"{el:.0f}s ({el/len(run):.2f}s/topic)")
    del rr
    import torch
    torch.cuda.empty_cache()
    return out, el


def recall_at(run: dict, qrels: dict, k: int) -> float:
    hit = tot = 0
    for tid, docs in qrels.items():
        gold = {d for d, r in docs.items() if r == data.ELIGIBLE}
        if not gold:
            continue
        top = set(sorted(run.get(tid, {}), key=lambda d: -run[tid][d])[:k])
        hit += len(top & gold)
        tot += len(gold)
    return hit / max(tot, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", help="run dau vao (vd runs/hybrid.dev.txt)")
    ap.add_argument("--bench", help="chay ca ba model tren run nay")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--out")
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.self_test:
        print("Self-test ba reranker (lien quan phai cao hon khong lien quan):")
        ok = self_test(args.device)
        print("TAT CA DAT" if ok else "CO RERANKER KHONG DAT")
        return 0 if ok else 1

    src = args.bench or args.run
    if not src:
        ap.error("can --run, --bench hoac --self-test")
    if args.year == data.TEST_YEAR:
        print("!! TAP TEST 2022 — chi cham MOT LAN o Phase 11.", file=sys.stderr)

    topics = data.load_topics(args.year)
    qrels = data.load_qrels(args.year)
    base = run_io.read_run(src)

    r_at_depth = recall_at(base, qrels, args.depth)
    a0 = metrics.aggregate(metrics.evaluate(base, qrels))
    print(f"Dau vao {src}")
    print(f"  TRAN CUNG: Recall@{args.depth} (eligible) = {r_at_depth:.4f}")
    print(f"  Khong reranker nao vuot qua duoc con so nay.")
    print(f"  elig nDCG@10 truoc rerank = {a0['eligible/ndcg_cut_10']:.4f}\n")

    keys = list(MODELS) if args.bench else [args.model]
    rows = []
    for key in keys:
        print(f"== {key} ==")
        r, el = rerank_run(base, topics, key, args.depth, args.device)
        a = metrics.aggregate(metrics.evaluate(r, qrels))
        out = args.out or f"runs/rerank_{key}.dev.txt"
        run_io.write_run(out, r, f"rerank_{key}", depth=args.depth)
        rows.append({"model": key, "elig_ndcg10": a["eligible/ndcg_cut_10"],
                     "official_ndcg10": a["official/ndcg_cut_10"],
                     "contam10": a["elig/contamination_10"],
                     "seconds": el, "sec_per_topic": el / len(base), "run": out})
        print(f"  elig nDCG@10={a['eligible/ndcg_cut_10']:.4f} "
              f"(truoc: {a0['eligible/ndcg_cut_10']:.4f})  -> {out}\n")

    if args.bench:
        print("=" * 78)
        hdr = (f"{'model':10s} {'eligNDCG10':>11s} {'offNDCG10':>10s} {'contam10':>9s} "
               f"{'giay':>7s} {'s/topic':>8s}")
        print(hdr); print("-" * len(hdr))
        print(f"{'(truoc)':10s} {a0['eligible/ndcg_cut_10']:11.4f} "
              f"{a0['official/ndcg_cut_10']:10.4f} {a0['elig/contamination_10']:9.4f} "
              f"{'-':>7s} {'-':>8s}")
        for r in sorted(rows, key=lambda x: -x["elig_ndcg10"]):
            print(f"{r['model']:10s} {r['elig_ndcg10']:11.4f} {r['official_ndcg10']:10.4f} "
                  f"{r['contam10']:9.4f} {r['seconds']:7.0f} {r['sec_per_topic']:8.2f}")
        print(f"\nTran cung Recall@{args.depth} = {r_at_depth:.4f}")
        os.makedirs("results", exist_ok=True)
        p = f"results/_rerank_bench.{args.year}.json"
        json.dump({"recall_at_depth": r_at_depth, "before": a0["eligible/ndcg_cut_10"],
                   "rows": rows}, open(p, "w", encoding="utf-8"), indent=2)
        print(f"Da ghi {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
