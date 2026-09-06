#!/usr/bin/env bash
#
# Chay ca bac thang end-to-end bang MOT lenh:
#   retrieval (lexical -> dense -> hybrid) -> rerank -> reasoning -> generation
# va cham diem sau moi bac.
#
#   bash scripts/run_ladder.sh                        # bac 0-4, bac 5 chi UOC LUONG
#   bash scripts/run_ladder.sh --yes                  # chay that ca bac 5 (ton quota)
#   bash scripts/run_ladder.sh --rerank-model qwen3   # doi reranker
#   bash scripts/run_ladder.sh --rerank-model all     # quet ca ba reranker
#   bash scripts/run_ladder.sh --stages 1,2,3         # chi mot phan
#   bash scripts/run_ladder.sh --dry-run              # in lenh, khong chay
#   PY=python bash scripts/run_ladder.sh              # tren Kaggle (khong co .venv)
#
# Model doi duoc bang co dong lenh CHI CO reranker (medcpt | bge | qwen3 | all).
# Dense va reasoning co dinh — ly do ghi ngay tren khai bao cua chung ben duoi.
#
# BA DIEU SCRIPT NAY CO TINH LAM:
#
# 1. Ghi ra tien to RIENG (mac dinh "ladder"), khong dung lai ten cu. Cac file
#    results/rung1_lexical.json ... rung5_eligibility.json la ket qua TAP TEST
#    2022 da dong bang o giai doan 11 va duoc whitelist trong .gitignore — ghi de
#    chung la mat vinh vien mot thu khong tao lai duoc.
#
# 2. Bac 5 mac dinh CHI chay --estimate. Suy luan la buoc duy nhat ton tien; neu
#    cache data/reasoning/ da du thi uoc luong se bao 0 lenh goi va ban co the
#    --yes yen tam. Chay thang vao API ma khong biet truoc so lenh goi la cach
#    tot nhat de dot sach quota trong mot dem.
#
# 3. Nam 2022 bi TU CHOI tru khi co --i-am-sure-test-set. Tap test da cham dung
#    mot lan. Canh bao in ra stderr trong bm25.py/score.py rat de troi qua giua
#    hang tram dong log cua mot lenh chay tu dong.

set -euo pipefail

PY=${PY:-.venv/bin/python}
export PYTHONPATH=${PYTHONPATH:-.}

YEAR=2021
PREFIX=ladder
DEVICE=cuda
STAGES=0,1,2,3,4,5
RERANK_MODEL=medcpt
LEXMODE=prof_narr
TOP_N=20

# Hai model KHONG phai co dong lenh, va do la co y:
#   dense — chi qwen3.base.npz la full corpus (375.580 doc); moi .npz khac deu la
#     mau con 46.162 doc. Doi ten model dense khong phai doi mot co, phai encode
#     lai ca corpus (~3,5h GPU).
#   reasoning — cache_path() la {year}.{model}.{mode}.json, nen doi model la sinh
#     cache moi hoan toan: vut 1.499 cap da suy luan va goi lai tu dau.
# Chung dung o day de banner in ra duoc va de dinh vi dung file profile/cache.
DENSE_MODEL=qwen3
DENSE_VECS=indexes/dense/qwen3.base.npz
REASON_MODEL=gemini-3.5-flash-lite
LEX_INDEX=indexes/bm25-critfields
FORCE=0
YES=0
DRY=0
SURE_TEST=0

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --year)     YEAR=$2; shift 2 ;;
    --prefix)   PREFIX=$2; shift 2 ;;
    --device)   DEVICE=$2; shift 2 ;;
    --stages)   STAGES=$2; shift 2 ;;
    --rerank-model) RERANK_MODEL=$2; shift 2 ;;
    --model)    RERANK_MODEL=$2; shift 2 ;;   # bi danh cua --rerank-model
    --lexical)  LEXMODE=$2; shift 2 ;;
    --top-n)    TOP_N=$2; shift 2 ;;
    --force)    FORCE=1; shift ;;
    --yes)      YES=1; shift ;;
    --dry-run)  DRY=1; shift ;;
    --i-am-sure-test-set) SURE_TEST=1; shift ;;
    -h|--help)  usage ;;
    *) echo "Tham so la: $1" >&2; exit 2 ;;
  esac
done

if [ "$YEAR" = "2022" ] && [ "$SURE_TEST" != "1" ]; then
  cat >&2 <<'EOF'
TU CHOI: 2022 la TAP TEST, da cham dung MOT LAN o giai doan 11.
Cham lai bang bat ky cau hinh nao cung lam hong tinh gia tri cua con so da bao cao.
Neu that su co ly do, them --i-am-sure-test-set.
EOF
  exit 3
fi

SPLIT=dev; [ "$YEAR" = "2022" ] && SPLIT=test

# Tag mang luon split, neu khong mot lan chay 2022 se ghi de ket qua dev cung ten
# trong results/ trong khi file run lai tach ra dung — sai lech im lang.
T1=${PREFIX}1_bm25.${SPLIT};    R1=runs/${T1}.txt
T2=${PREFIX}2_dense.${SPLIT};   R2=runs/${T2}.txt
T3=${PREFIX}3_hybrid.${SPLIT};  R3=runs/${T3}.txt
T5=${PREFIX}5_elig.${SPLIT};    R5=runs/${T5}.txt

# Ten bac 4 mang luon ten reranker, ke ca voi model mac dinh. Khong co no thi
# chay medcpt roi chay bge se ghi de len nhau trong ca runs/ lan results/ ma
# khong bao gi — ban van co mot bang so, chi la sai. Cac bac khac giu nguyen ten
# vi doi reranker khong lam chung thay doi.
#
# RERANK_MODEL co the la mot duong dan (/kaggle/working/... hoac to-chuc/ten-model),
# nen phai rut gon thanh slug: de nguyen thi dau '/' bien thanh thu muc va
# `run_io.write_run` se hong vi thu muc do khong ton tai.
RR_SLUG=$(printf '%s' "$RERANK_MODEL" | sed 's#/*$##; s#.*/##; s#[^A-Za-z0-9._-]#_#g')
T4=${PREFIX}4_rerank-${RR_SLUG}.${SPLIT};  R4=runs/${T4}.txt
mkdir -p runs results logs

LOG=logs/${PREFIX}_${YEAR}_$(date +%Y%m%d-%H%M%S).log

has()  { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
skip() { [ "$FORCE" = 0 ] && [ -s "$1" ]; }

run() {
  printf '   $ %s\n' "$*"
  [ "$DRY" = 1 ] && return 0
  "$@"
}

# ---------------------------------------------------------------- preflight
say "Preflight"
miss=0
for p in data/trials.db indexes/bm25-critfields \
         indexes/dense/qwen3.base.npz \
         rawdata/topics${YEAR}.xml rawdata/qrels${YEAR}.txt; do
  if [ -e "$p" ]; then printf '   CO    %s\n' "$p"
  else printf '   THIEU %s\n' "$p"; miss=1; fi
done
[ "$miss" = 1 ] && { echo "Thieu dau vao, dung lai." >&2; exit 4; }
$PY - <<'EOF'
import sqlite3
n = sqlite3.connect("data/trials.db").execute(
    "select count(*) from trials").fetchone()[0]
print(f"   trials.db: {n:,} trial" + ("" if n == 375_580 else "  << KHONG khop 375,580"))
EOF
echo "   log: $LOG"

exec > >(tee -a "$LOG") 2>&1

# ------------------------------------------------------------------- cau hinh
# Mot lan chay tu dong dai hang tram dong; khong co banner nay thi doc lai log cu
# khong con biet da chay model nao.
say "Cau hinh"
printf '   %-11s %-22s %s\n' \
  "lexical"   "$LEXMODE"      "$LEX_INDEX  k1=1.8 b=1.0" \
  "dense"     "$DENSE_MODEL"  "$DENSE_VECS" \
  "rerank"    "$RERANK_MODEL" "depth 100" \
  "reasoning" "$REASON_MODEL" "top-n $TOP_N   cache data/reasoning/" \
  "output"    "$PREFIX*"      "runs/ + results/  (nam $YEAR, $SPLIT)"

# ------------------------------------------------------------------- bac 0
# Trich ho so benh nhan. Chan lexical prof_narr doc file nay; no da co san trong
# repo nen bac 0 hau nhu khong bao gio chay that — no ton tai cho may sach.
if has 0; then
  say "Bac 0 — extraction (ho so benh nhan)"
  PROF=data/profiles/${YEAR}.${REASON_MODEL}.json
  if [ -s "$PROF" ]; then
    echo "   bo qua, da co $PROF"
  elif [ "$YES" != 1 ]; then
    echo "   THIEU $PROF va trich lai ton lenh goi Gemini — them --yes de chay."
    echo "   Bac 1 se hong neu --lexical van la $LEXMODE."
  else
    run $PY -m src.extraction.extract --year "$YEAR" --model "$REASON_MODEL"
  fi
fi

# ------------------------------------------------------------------- bac 1
# Chan lexical THAT cua du an la truy van da mo rong bang ho so + narrative
# (src.extraction.query --mode prof_narr), khong phai topic tho. Do bang so tren
# dev 2021: topic tho cho hybrid eligN@10 0.3264, prof_narr cho 0.3501. Dung
# bm25.py truc tiep la tu bo 0.024 ma khong nhan lai gi.
# Doc ho so da trich san trong data/profiles/ nen KHONG ton lenh goi API.
# --lexical raw de doi chieu khi can.
if has 1; then
  say "Bac 1 — lexical (BM25 critfields, truy van $LEXMODE)"
  if skip "$R1"; then echo "   bo qua, da co $R1"; else
    if [ "$LEXMODE" = raw ]; then
      run $PY -m src.retrieval.bm25 --index indexes/bm25-critfields \
          --k1 1.8 --b 1.0 --year "$YEAR" --out "$R1" --tag "$T1"
    else
      run $PY -m src.extraction.query --mode "$LEXMODE" --year "$YEAR" --out "$R1"
    fi
  fi
  run $PY -m src.eval.score "$R1" --year "$YEAR" --tag "$T1"
fi

# ------------------------------------------------------------------- bac 2
if has 2; then
  say "Bac 2 — dense (qwen3)"
  if skip "$R2"; then echo "   bo qua, da co $R2"; else
    run $PY -m src.dense.search --model qwen3 \
        --vecs indexes/dense/qwen3.base.npz \
        --year "$YEAR" --device "$DEVICE" --out "$R2"
  fi
  run $PY -m src.eval.score "$R2" --year "$YEAR" --tag "$T2" \
      --vs "results/${T1}.json"
fi

# ------------------------------------------------------------------- bac 3
if has 3; then
  say "Bac 3 — hybrid (RRF)"
  if skip "$R3"; then echo "   bo qua, da co $R3"; else
    run $PY -m src.retrieval.fusion --lexical "$R1" --dense "$R2" \
        --out "$R3" --year "$YEAR"
  fi
  run $PY -m src.eval.score "$R3" --year "$YEAR" --tag "$T3" \
      --vs "results/${T2}.json"
fi

# ------------------------------------------------------------------- bac 4
# Rerank cat con top-100, nen Recall@1000 cua bac nay KHONG so sanh duoc voi
# bac 3 — xem chu thich † trong README.
if has 4; then
  say "Bac 4 — + reranking ($RERANK_MODEL)"
  if [ "$RERANK_MODEL" = all ]; then
    # rerank.py --bench chay ca ba model va tu ghi results/_rerank_bench.{year}.json
    # kem seconds/sec_per_topic. KHONG truyen --out: voi --bench thi
    # out = args.out or runs/rerank_{key}.dev.txt, nen mot --out se khien ca ba
    # model ghi de len cung mot file.
    run $PY -m src.rerank.rerank --bench "$R3" --year "$YEAR" --device "$DEVICE"
  else
    if skip "$R4"; then echo "   bo qua, da co $R4"; else
      run $PY -m src.rerank.rerank --run "$R3" --model "$RERANK_MODEL" \
          --out "$R4" --year "$YEAR" --device "$DEVICE"
    fi
    run $PY -m src.eval.score "$R4" --year "$YEAR" --tag "$T4" \
        --vs "results/${T3}.json"
  fi
fi

# ------------------------------------------------------------------- bac 5
# Xep lai tren HYBRID chu khong tren rerank: bac 5 giu nguyen do sau truy hoi
# cua bac 3 (Recall@1000 khong doi), dung nhu bang ket qua trong README.
if has 5; then
  say "Bac 5 — + eligibility reasoning"
  echo "   Uoc luong chi phi truoc khi goi API:"
  run $PY -m src.reasoning.reason --run "$R3" --year "$YEAR" \
      --top-n "$TOP_N" --estimate

  if [ "$YES" != 1 ]; then
    cat <<EOF

   DUNG O DAY. Tren la so lenh goi Gemini se ton neu chay that.
   Cache co san trong data/reasoning/ duoc dung lai, nen con so nay co the la 0.
   Chay that:  bash $0 --stages 5 --yes
EOF
  else
    run $PY -m src.reasoning.reason --run "$R3" --year "$YEAR" --top-n "$TOP_N"
    run $PY -m src.reasoning.score --year "$YEAR" --rule strict \
        --base-run "$R3" --emit "$R5" --vs "results/${T3}.json"
    run $PY -m src.eval.score "$R5" --year "$YEAR" --tag "$T5" \
        --vs "results/${T3}.json"
  fi
fi

# --------------------------------------------------------------- generation
# Giai doan 9: danh bong bang LLM + kiem tra trich dan tu dong. Do la 20 lenh
# goi rieng (~25 phut do duoc), khong nam trong uoc luong cua bac 5.
if has gen; then
  say "Generation — giai doan 9 (danh bong + kiem tra trich dan)"
  if [ "$YES" != 1 ]; then
    echo "   Can --yes: buoc nay goi them ~20 lenh Gemini (~25 phut)."
  else
    run $PY scripts/phase9_generate_samples.py
    run $PY scripts/phase9_assemble_reports.py
    run $PY scripts/phase9_citation_check.py results/_phase9_reports.dev.json
  fi
fi

# ------------------------------------------------------------------ tong ket
say "Tong ket"
[ "$DRY" = 1 ] || $PY - "$PREFIX" "$SPLIT" <<'EOF'
import glob, json, os, sys
prefix, split = sys.argv[1], sys.argv[2]


def add(rows, name, path):
    if not os.path.exists(path):
        return
    a = json.load(open(path))["aggregate"]
    rows.append((name, a["eligible/ndcg_cut_10"], a["official/ndcg_cut_10"],
                 a["elig/recall_1000"], a["elig/contamination_10"]))


rows = []
add(rows, "1 lexical", f"results/{prefix}1_bm25.{split}.json")
add(rows, "2 dense", f"results/{prefix}2_dense.{split}.json")
add(rows, "3 hybrid", f"results/{prefix}3_hybrid.{split}.json")
# Moi bien the reranker deu hien thanh mot dong rieng, nen mot lan quet model
# doc duoc ngay canh nhau thay vi phai mo tung file results.
for p in sorted(glob.glob(f"results/{prefix}4_rerank-*.{split}.json")):
    key = os.path.basename(p).split("4_rerank-", 1)[1].rsplit(f".{split}.", 1)[0]
    add(rows, f"4 rerank {key}", p)
add(rows, "5 eligibility", f"results/{prefix}5_elig.{split}.json")
if not rows:
    print("   chua co ket qua nao.")
else:
    print(f"   {'bac':16s} {'eligN@10':>9s} {'offN@10':>9s} "
          f"{'R@1000':>8s} {'contam@10':>10s}")
    for r in rows:
        print(f"   {r[0]:16s} {r[1]:9.4f} {r[2]:9.4f} {r[3]:8.4f} {r[4]:10.4f}")
    print("\n   contam@10 doc CUNG LUC voi eligN@10: mot cau hinh cat o nhiem"
          "\n   bang cach loai bot thu nghiem khong phai la mot cai tien.")
EOF
echo
echo "Log day du: $LOG"
