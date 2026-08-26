-- TrialMatch-RAG — canonical record store (Phase 1)
--
-- Nguon su that duy nhat sau khi parse. Sau buoc nay khong doc lai XML nua.
-- Lexical index (Lucene) va vector index (Qdrant) deu duoc build TU bang nay,
-- khong phai tu rawdata/.

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS trials (
    nct_id             TEXT PRIMARY KEY,
    title              TEXT,
    summary            TEXT,
    detail             TEXT,

    -- Bo loc co cau truc. NULL = trial khong khai bao (KHONG phai 0, khong phai
    -- "khong gioi han"). Invariant 3: thieu thong tin phai giu nguyen la thieu.
    gender             TEXT,
    min_age_years      REAL,
    max_age_years      REAL,
    min_age_raw        TEXT,
    max_age_raw        TEXT,
    healthy_volunteers TEXT,

    phase              TEXT,
    status             TEXT,
    study_type         TEXT,

    -- Blob eligibility da normalize. BAT BUOC phai luu: cap span_start/span_end
    -- trong bang criteria tro vao chinh chuoi nay. Khong co no thi offset vo
    -- nghia va Phase 8 mat kha nang KIEM CHUNG trich dan cua LLM.
    criteria_raw       TEXT,

    -- Chat luong parse, dung cho bao cao Phase 1 va de loc ve sau.
    parse_method       TEXT,     -- bulleted | numbered | mixed | sentence_split | none
    n_criteria         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS criteria (
    nct_id     TEXT    NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    section    TEXT    NOT NULL CHECK (section IN ('inclusion', 'exclusion', 'unknown')),
    text       TEXT    NOT NULL,
    -- Offset vao trials.criteria_raw. Bat buoc thoa:
    --   criteria_raw[span_start:span_end] chua text nguyen van.
    span_start INTEGER NOT NULL,
    span_end   INTEGER NOT NULL,
    -- Cau dan xuat hien truoc danh sach, vd "Acute onset of:" dung truoc cac
    -- muc danh so. Tach roi thi tung muc mat nghia, nen giu lai lam ngu canh.
    lead_in    TEXT,
    PRIMARY KEY (nct_id, idx)
);

-- Truong lap lai. Tach bang rieng thay vi nhoi JSON vao trials, vi Phase 11
-- can aggregate theo condition/mesh de phan tich loi.
CREATE TABLE IF NOT EXISTS trial_conditions    (nct_id TEXT NOT NULL, term TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trial_interventions (nct_id TEXT NOT NULL, term TEXT NOT NULL, itype TEXT);
CREATE TABLE IF NOT EXISTS trial_keywords      (nct_id TEXT NOT NULL, term TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trial_mesh          (nct_id TEXT NOT NULL, term TEXT NOT NULL, source TEXT);

CREATE INDEX IF NOT EXISTS ix_criteria_nct     ON criteria(nct_id);
CREATE INDEX IF NOT EXISTS ix_criteria_section ON criteria(section);
CREATE INDEX IF NOT EXISTS ix_cond_nct         ON trial_conditions(nct_id);
CREATE INDEX IF NOT EXISTS ix_cond_term        ON trial_conditions(term);
CREATE INDEX IF NOT EXISTS ix_intv_nct         ON trial_interventions(nct_id);
CREATE INDEX IF NOT EXISTS ix_kw_nct           ON trial_keywords(nct_id);
CREATE INDEX IF NOT EXISTS ix_mesh_nct         ON trial_mesh(nct_id);
CREATE INDEX IF NOT EXISTS ix_mesh_term        ON trial_mesh(term);
