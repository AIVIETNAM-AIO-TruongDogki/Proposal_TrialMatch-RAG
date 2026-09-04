"""Phase 8 buoc 6 — cham diem ba trang thai, va bac 5 tren thang ablation.

    python -m src.reasoning.score --year 2021
    python -m src.reasoning.score --year 2021 --rule lenient --emit runs/elig.dev.txt

MACRO-F1 MOT MINH LA GAMEABLE — LUON BAO CAO KEM HAI SO KIA
------------------------------------------------------------
Mot model tra loi `unverifiable` cho MOI tieu chi van co the dat macro-F1
trong coi duoc, trong khi vo dung hoan toan. Vi vay bo ba luon di cung nhau:

  * macro-F1 tren {satisfied, violated, unverifiable}
  * TY LE KIENG (abstention) — bao nhieu phan tram tra loi `unverifiable`
  * do chinh xac tren phan KHONG kieng

Doc ba so nay cung luc thi khong the giau mot model chi biet noi "khong biet".

KHONG CO NHAN VANG MUC TIEU CHI — NOI THANG DIEU DO
---------------------------------------------------
qrels cua TREC chi co nhan muc TRIAL (0/1/2). Khong the cham truc tiep tung
tieu chi. Nen o day dung duong (a) cua specs/08: gop quyet dinh tieu chi thanh
quyet dinh trial roi doi chieu qrels. He qua phai noi ro: phep do nay do CA
luat gop lan chat luong suy luan, khong tach duoc hai thu. Muon tach thi can
bo nhan tu gan tay (duong (c)), la deliverable vuon toi chu chua co.

CON SO TIEU DE LA CONTAMINATION@10, KHONG PHAI nDCG
----------------------------------------------------
Bac 5 co the lam nDCG chinh thuc GIAM ma van dung: qrels chinh thuc cho diem
duong cho trial `excluded`, nen mot bo loc eligibility hoat dong TOT se bi tru
diem o ho do do. Cai phai giam la contamination@10.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from src.eval import data, metrics, run_io, sig
from src.reasoning import aggregate, reason, schema


def load_decisions(path: str, want_hash: str | None = None) -> dict:
    """Doc mot file quyet dinh. `want_hash` KHONG phai tuy chon khi so sanh.

    `reason.load_cache` kiem `prompt_hash` truoc khi dung lai cache, nhung ham
    nay truoc day thi khong — nen mot file cu con sot lai van duoc nap va dem
    ra so sanh nhu that. Do la dieu vua xay ra: ablation lua chon ep buoc doc
    phai ban 30/08 (hash ceb71ba6c6cb, DUY NHAT 1 ban ghi, prompt cu) va in ra
    "hai trang thai F1=0.0000 acc=1.0000" — mot cap so vo nghia ma van trong
    nhu ket qua. So sanh hai file sinh boi hai prompt khac nhau khong do duoc
    gi ca.
    """
    blob = json.load(open(path, encoding="utf-8"))
    got = blob.get("prompt_hash")
    if want_hash and got != want_hash:
        raise SystemExit(
            f"Khong the so sanh: {path}\n"
            f"  prompt_hash = {got}, can {want_hash}.\n"
            f"  File nay sinh boi prompt/schema KHAC — dem ra so sanh la do hai\n"
            f"  thi nghiem khac nhau. Chay lai no voi prompt hien tai, hoac xoa di.")
    out = {}
    for key, rec in blob["records"].items():
        tid, nct = key.split("|", 1)
        out[(tid, nct)] = rec["decisions"]
    return out, blob


def label_stats(decisions_by: dict) -> dict:
    c = Counter(d["label"] for ds in decisions_by.values() for d in ds)
    tot = sum(c.values())
    return {"counts": dict(c), "total": tot,
            "abstention": c["unverifiable"] / tot if tot else 0.0}


def trial_level_eval(decisions_by: dict, qrels: dict, rule: str) -> dict:
    """Gop len muc trial roi doi chieu qrels (duong (a) cua specs/08).

    Nhan du doan: eligible neu diem gop > 0 (khong bi luat gop loai).
    Nhan vang: ELIGIBLE(2) = 1, con lai = 0 — cung ban do voi eligible_only().
    """
    tp = fp = fn = tn = 0
    for (tid, nct), ds in decisions_by.items():
        gold = qrels.get(tid, {}).get(nct)
        if gold is None:
            continue                      # ngoai pool da cham, khong ket luan duoc
        y = 1 if gold == data.ELIGIBLE else 0
        p = 1 if aggregate.trial_score(ds, rule) > 0 else 0
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif not p and y:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n_judged": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / n if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=data.DEV_YEAR, choices=[2021, 2022])
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="trial", choices=["trial", "criterion"])
    ap.add_argument("--rule", default="strict", choices=list(aggregate.RULES))
    ap.add_argument("--all-rules", action="store_true",
                    help="ablate ca ba luat gop canh nhau")
    ap.add_argument("--base-run", default="runs/bm25_best.dev.txt")
    ap.add_argument("--emit", default=None, help="ghi run da xep lai")
    ap.add_argument("--vs", default="results/bm25_best.dev.json")
    ap.add_argument("--out-dir", default=reason.OUT_DIR)
    args = ap.parse_args()

    from src.extraction import gemini
    model = args.model or gemini.MODEL

    path = reason.cache_path(args.year, model, args.mode, False, args.out_dir)
    forced_path = reason.cache_path(args.year, model, args.mode, True, args.out_dir)
    if not os.path.exists(path):
        print(f"Chua co {path}. Chay src.reasoning.reason truoc.", file=sys.stderr)
        return 1

    dec, blob = load_decisions(path)
    qrels = data.load_qrels(args.year)
    base = run_io.read_run(args.base_run)

    st = label_stats(dec)
    print(f"{len(dec):,} cap (topic,trial), {st['total']:,} quyet dinh da qua "
          f"kiem chung grounding")
    print(f"\nPHAN BO NHAN")
    for lab in schema.LABELS:
        n = st["counts"].get(lab, 0)
        print(f"  {lab:14s} {n:6,}  {n/max(st['total'],1)*100:5.1f}%")
    print(f"\n  TY LE KIENG (unverifiable) = {st['abstention']*100:.1f}%")
    print("  Doc con nay CUNG LUC voi F1 ben duoi: mot model tra loi")
    print("  'unverifiable' cho tat ca van co F1 coi duoc ma vo dung.")

    rules = list(aggregate.RULES) if args.all_rules else [args.rule]
    print(f"\nMUC TRIAL (gop tu tieu chi, doi chieu qrels {args.year})")
    hdr = f"{'luat':10s} {'n':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} {'acc':>7s}"
    print(hdr); print("-" * len(hdr))
    for rule in rules:
        e = trial_level_eval(dec, qrels, rule)
        print(f"{rule:10s} {e['n_judged']:6,} {e['precision']:7.4f} "
              f"{e['recall']:7.4f} {e['f1']:7.4f} {e['accuracy']:7.4f}")
    print("\nLuu y: phep do nay do CA luat gop lan chat luong suy luan — qrels")
    print("chi co nhan muc trial, khong tach duoc hai thu (specs/08 duong (a)).")

    if os.path.exists(forced_path):
        # KHONG so voi blob['prompt_hash'] cua file KHONG-ep-buoc — hai file co
        # prompt_hash KHAC NHAU CO CHU DICH (schema ep buoc bo nhan `unverifiable`
        # khoi enum, xem schema.decision_schema(forced=True)). Phai tinh rieng
        # hash MONG DOI cua chinh file ep buoc.
        want_forced_hash = schema.prompt_hash(True, args.mode == "trial")
        fdec, _ = load_decisions(forced_path, want_forced_hash)
        if len(fdec) < len(dec):
            print(f"\n(Ablation ep buoc CHUA DAY DU: {len(fdec):,}/{len(dec):,} cap "
                  f"— chay --forced cho het truoc khi doc con so nay)")
        fe = trial_level_eval(fdec, qrels, args.rule)
        e = trial_level_eval(dec, qrels, args.rule)
        print(f"\nABLATION LUA CHON EP BUOC (kiem chung invariant 1)")
        print(f"  ba trang thai  F1={e['f1']:.4f}  acc={e['accuracy']:.4f}")
        print(f"  hai trang thai F1={fe['f1']:.4f}  acc={fe['accuracy']:.4f}")
        if fe["f1"] >= e["f1"]:
            print("  -> Hai trang thai lam KHONG KEM HON. Dong gop ba trang thai")
            print("     CHUA duoc chung minh tren du lieu nay — phai noi ro.")
    else:
        print(f"\n(Chua co ablation ep buoc: chay --forced de sinh {forced_path})")

    if args.emit:
        r = aggregate.rerank_by_eligibility(base, dec, args.rule)
        run_io.write_run(args.emit, r, f"elig_{args.rule}")
        a_new = metrics.aggregate(metrics.evaluate(r, qrels))
        a_old = metrics.aggregate(metrics.evaluate(base, qrels))
        print(f"\nBAC 5 vs BAC DUOI ({args.base_run})")
        print(f"  contamination@10  {a_old['elig/contamination_10']:.4f} -> "
              f"{a_new['elig/contamination_10']:.4f}   <- CON SO TIEU DE")
        print(f"  elig nDCG@10      {a_old['eligible/ndcg_cut_10']:.4f} -> "
              f"{a_new['eligible/ndcg_cut_10']:.4f}")
        print(f"  official nDCG@10  {a_old['official/ndcg_cut_10']:.4f} -> "
              f"{a_new['official/ndcg_cut_10']:.4f}")
        print("  (official co the GIAM ma van dung: qrels chinh thuc cho diem")
        print("   duong cho trial 'excluded' — xem docstring metrics.py)")
        print(f"\nDa ghi {args.emit}. Cham day du:")
        print(f"  PYTHONPATH=. .venv/bin/python -m src.eval.score {args.emit} "
              f"--year {args.year} --vs {args.vs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
