"""API doc canonical store.

Moi tang phia sau (retrieval, reranking, reasoning, generation) doc trial qua
day, khong doc lai XML. Sau Phase 1 thi `rawdata/` va chi muc nct_id -> path
deu khong con duoc dung luc chay nua.
"""

from __future__ import annotations

import sqlite3


def open_db(path: str = "data/trials.db", check_same_thread: bool = True
           ) -> sqlite3.Connection:
    """`check_same_thread=False` — can cho Phase 10: server song goi cac ham
    doc-chi (get_trial/get_criteria/...) tu nhieu asyncio.to_thread() dong
    thoi tren MOT connection. SQLite bien dich mac dinh o che do "serialized"
    (SQLITE_THREADSAFE=1) nen ban than thu vien da tu khoa noi bo — chia se
    mot connection giua nhieu thread la AN TOAN ve du lieu, chi mat song song
    that su (cac lenh xep hang cho nhau), khong dang ke voi truy van diem theo
    khoa chinh nhu o day. Cac script batch (mot tien trinh, mot luong) khong
    doi hanh vi mac dinh.
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
    """Van ban dung de index (lexical + dense).

    CHU Y: khong gop criteria_raw vao day. Text criteria rat dai va day phu
    dinh, de nhan chim tin hieu chu de. Neu muon gop thi phai lam thanh mot
    run RIENG de ablate o Phase 5/6, khong sua ham nay.
    """
    parts = [trial.get("title"), trial.get("summary")]
    for key in ("conditions", "interventions", "mesh", "keywords"):
        if trial.get(key):
            parts.append("; ".join(trial[key]))
    return "\n".join(p for p in parts if p)


def verify_quote(conn: sqlite3.Connection, nct_id: str, idx: int, quote: str) -> bool:
    """Cau `quote` do LLM trich co that su nam trong criterion do khong?

    Day la cach bien invariant 3 tu mot loi hua thanh mot phep do. Phase 8
    goi ham nay tren moi criterion_quote; ty le that bai chinh la ty le
    ungrounded, va no la mot con so bao cao duoc.

    So khop sau khi chuan hoa khoang trang, vi criterion goc bi wrap cung
    con LLM tra ve mot dong lien.
    """
    row = conn.execute(
        "SELECT text FROM criteria WHERE nct_id = ? AND idx = ?", (nct_id, idx)
    ).fetchone()
    if row is None:
        return False
    norm = lambda s: " ".join(s.split()).lower()
    return norm(quote) in norm(row[0])


def criterion_source(conn: sqlite3.Connection, nct_id: str, idx: int) -> str | None:
    """Doan van goc (chua chuan hoa khoang trang) ma criterion nay duoc cat ra.

    Dung khi can trung bay bang chung nguyen ban cho nguoi doc kiem tra.
    """
    row = conn.execute(
        "SELECT t.criteria_raw, c.span_start, c.span_end "
        "FROM criteria c JOIN trials t USING(nct_id) "
        "WHERE c.nct_id = ? AND c.idx = ?", (nct_id, idx)
    ).fetchone()
    return row[0][row[1]:row[2]] if row else None
