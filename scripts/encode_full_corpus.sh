#!/usr/bin/env bash
# Phase 5 buoc cuoi — ma hoa TOAN CORPUS (375.580 trial) bang cau hinh thang cuoc.
#
#     bash scripts/encode_full_corpus.sh
#
# ~212 phut. Moi shard MOT TIEN TRINH RIENG (GPU sach giua cac luot) va duoc luu
# ngay khi xong, nen dut giua chung chi mat shard dang chay — chay lai script se
# bo qua cac shard da co.
set -u
cd "$(dirname "$0")/.." || exit 1

# Thieu o lan chay dau: gay canh bao OOM lap lai (retry, khong crash) va lam
# toc do tut tu 52 xuong 24 chunk/s do phan manh GPU tich luy qua thoi gian.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=qwen3
VARIANT=base            # union-recall chon `fields`; chon `base` la SAI LECH co
                        # ghi nhan — xem docs/decisions/phase5-dense.md
OUT=indexes/dense/${MODEL}.${VARIANT}.npz

command -v ollama >/dev/null && ollama stop qwen3:8b 2>/dev/null
echo "GPU truoc khi bat dau:"; nvidia-smi --query-gpu=memory.used --format=csv,noheader

for n in 0 1 2 3; do
  echo "########## shard $n  ($(date +%H:%M:%S)) ##########"
  PYTHONPATH=. .venv/bin/python -u -m src.dense.encode \
      --model "$MODEL" --input "data/jsonl/${VARIANT}" --out "$OUT" --shard "$n" \
    2>&1 | grep --line-buffered -vE "Loading weights|it/s\]$"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!! shard $n loi rc=$rc — DUNG LAI. Chay lai script de tiep tuc."
    exit "$rc"
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done

echo "########## gop shard  ($(date +%H:%M:%S)) ##########"
PYTHONPATH=. .venv/bin/python -m src.dense.encode \
    --model "$MODEL" --out "$OUT" --merge || exit 1

echo "########## tim kiem + cham diem  ($(date +%H:%M:%S)) ##########"
PYTHONPATH=. .venv/bin/python -u -m src.dense.search \
    --vecs "$OUT" --model "$MODEL" --out runs/dense.dev.txt \
  2>&1 | grep -vE "Loading weights|it/s\]$" || exit 1

echo "########## KET QUA — lan dau so sanh HOP LE voi Phase 3 ##########"
PYTHONPATH=. .venv/bin/python - <<'PY'
import statistics as st
from src.eval import data, metrics, run_io, sig
q = data.load_qrels(2021)
runs = [("BM25 bac 1 (bm25_best)", "runs/bm25_best.dev.txt"),
        ("Dense bac 2 (qwen3 base)", "runs/dense.dev.txt")]
per = {}
for n, p in runs:
    r = run_io.read_run(p)
    a = metrics.aggregate(metrics.evaluate(r, q))
    per[n] = metrics.evaluate(r, q)
    per[n]["elig/recall_1000"] = metrics.eligible_recall(r, q, 1000)
    j = st.mean(metrics.judged_at_k(r, q, 10).values())
    c = st.mean(metrics.contamination_at_k(r, q, 10).values())
    print(f"{n:26s} off@10={a['official/ndcg_cut_10']:.4f} elig@10={a['eligible/ndcg_cut_10']:.4f} "
          f"rec@1k={a['elig/recall_1000']:.4f} contam={c:.4f} judged={j:.4f} chuanhoa={c/j:.4f}")
print("\nDense - BM25 (paired bootstrap, 75 benh an):")
A, B = per["Dense bac 2 (qwen3 base)"], per["BM25 bac 1 (bm25_best)"]
for m in ("eligible/ndcg_cut_10", "official/ndcg_cut_10", "elig/recall_1000"):
    r = sig.paired_bootstrap(A[m], B[m])
    tag = "CO Y NGHIA" if r["p"] < 0.05 else "ns"
    print(f"  {m:22s} {r['diff']:+.4f}  p={r['p']:.4f}  {tag}")
print("\nLUU Y: dense THUA BM25 van la ket qua hop le (specs/05), khong phai loi.")
PY
