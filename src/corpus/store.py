"""Canonical read store.

Every downstream stage (retrieval, reranking, reasoning, generation) reads
trials through here, never the raw XML — after Phase 1, `rawdata/` and the
nct_id -> path index are no longer touched at runtime.
"""

from __future__ import annotations

import sqlite3


def open_db(path: str = "data/trials.db", check_same_thread: bool = True
           ) -> sqlite3.Connection:
    """`check_same_thread=False` is for Phase 10: the live server calls
    read-only functions from multiple asyncio.to_thread() calls on one
    connection. SQLite's default "serialized" mode already locks internally,
    so sharing a connection across threads is data-safe — it only loses true
    parallelism (queries queue), negligible for point lookups by primary key.
    Batch scripts (one process, one thread) keep the default.
    """
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn


def get_trial(conn: sqlite3.Connection, nct_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM trials WHERE nct_id = ?", (nct_id,)).fetchone()
    if row is None:
        return None
    t = dict(row)
    for key, table, col in (
        ("conditions",    "trial_conditions",    "term"),
        ("interventions", "trial_interventions", "term"),
        ("keywords",      "trial_keywords",      "term"),
        ("mesh",          "trial_mesh",          "term"),
    ):
        t[key] = [r[0] for r in conn.execute(
            f"SELECT {col} FROM {table} WHERE nct_id = ?", (nct_id,))]
    return t


def get_criteria(conn: sqlite3.Connection, nct_id: str,
                 section: str | None = None) -> list[dict]:
    sql = "SELECT * FROM criteria WHERE nct_id = ?"
    args: list = [nct_id]
    if section:
        sql += " AND section = ?"
        args.append(section)
    return [dict(r) for r in conn.execute(sql + " ORDER BY idx", args)]


def retrieval_text(trial: dict) -> str:
    """Text used for indexing (lexical + dense).

    Deliberately excludes criteria_raw — criteria text is long and
    negation-heavy, drowning out topical signal. Including it should be a
    separate run to ablate in Phase 5/6, not a change to this function.
    """
    parts = [trial.get("title"), trial.get("summary")]
    for key in ("conditions", "interventions", "mesh", "keywords"):
        if trial.get(key):
            parts.append("; ".join(trial[key]))
    return "\n".join(p for p in parts if p)


def verify_quote(conn: sqlite3.Connection, nct_id: str, idx: int, quote: str) -> bool:
    """Is the LLM-extracted `quote` actually inside that criterion's text?

    Turns invariant 3 from a promise into a measurement: Phase 8 calls this
    on every criterion_quote, and the failure rate is the ungrounded rate, a
    reportable number. Matched after whitespace normalization, since the
    source criterion is wrapped while the LLM returns one line.
    """
    row = conn.execute(
        "SELECT text FROM criteria WHERE nct_id = ? AND idx = ?", (nct_id, idx)
    ).fetchone()
    if row is None:
        return False
    norm = lambda s: " ".join(s.split()).lower()
    return norm(quote) in norm(row[0])


def criterion_source(conn: sqlite3.Connection, nct_id: str, idx: int) -> str | None:
    """The raw (unnormalized) span this criterion was cut from.

    Used to display the original evidence text for a reader to check.
    """
    row = conn.execute(
        "SELECT t.criteria_raw, c.span_start, c.span_end "
        "FROM criteria c JOIN trials t USING(nct_id) "
        "WHERE c.nct_id = ? AND c.idx = ?", (nct_id, idx)
    ).fetchone()
    return row[0][row[1]:row[2]] if row else None
