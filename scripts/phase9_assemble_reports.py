"""Phase 9 — collapse the polished output into the reader-facing report.

    PYTHONPATH=. .venv/bin/python scripts/phase9_assemble_reports.py

Pure post-processing of results/_phase9_samples.dev.json: no model call, so the
prose is unchanged and only its presentation is. Writes
results/_phase9_reports.dev.json, which the citation check audits the same way:

    PYTHONPATH=. .venv/bin/python scripts/phase9_citation_check.py \\
        results/_phase9_reports.dev.json
"""

from __future__ import annotations

import json
import statistics as st

from src.generation import polish

SAMPLES = "results/_phase9_samples.dev.json"
OUT = "results/_phase9_reports.dev.json"


def main() -> int:
    blob = json.load(open(SAMPLES, encoding="utf-8"))
    out = []
    for s in blob["samples"]:
        out.append({**s, "prose": polish.assemble(s["card"], s["prose"])})

    before = [len(s["prose"]["claims"]) for s in blob["samples"]]
    after = [len(s["prose"]["claims"]) for s in out]
    print(f"{len(out)} bao cao\n")
    print(f"  so cau      truoc: trung vi {st.median(before):.0f}, max {max(before)}")
    print(f"              sau:   trung vi {st.median(after):.0f}, max {max(after)}")
    print(f"  >15 cau     truoc: {sum(1 for x in before if x > 15)}/20"
          f"   sau: {sum(1 for x in after if x > 15)}/20")
    print(f"  cau hoi mo  truoc: max {max(len(s['prose'].get('open_questions') or []) for s in blob['samples'])}"
          f"   sau: max {max(len(s['prose']['open_questions']) for s in out)}")

    json.dump({**{k: v for k, v in blob.items() if k != "samples"}, "samples": out},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nDa ghi {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
