"""Doc/ghi run file dinh dang TREC + ghi so ket qua.

Dinh dang TREC:  `topic_id Q0 doc_id rank score run_tag`

Moi lan chay deu duoc ghi lai kem commit git va cau hinh. Khong co so ghi thi
den Phase 11 se khong the tai lap hay giai thich noi mot con so nao.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

Run = dict[str, dict[str, float]]


def write_run(path: str, run: Run, tag: str, depth: int = 1000) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for tid in sorted(run):
            ranked = sorted(run[tid].items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
            for rank, (doc, score) in enumerate(ranked, 1):
                fh.write(f"{tid} Q0 {doc} {rank} {score:.6f} {tag}\n")


def read_run(path: str) -> Run:
    run: Run = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            run.setdefault(p[0], {})[p[2]] = float(p[4])
    return run


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def log_result(results_dir: str, tag: str, agg: dict[str, float],
               per_topic: dict[str, dict[str, float]], config: dict) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "tag": tag,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "config": config,
            "aggregate": agg,
            "per_topic": per_topic,
        }, fh, indent=2, sort_keys=True)
    return path


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
