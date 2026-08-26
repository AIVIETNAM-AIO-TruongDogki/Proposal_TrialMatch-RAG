"""Parse ClinicalTrials.gov XML thanh record + criteria da tach.

Hai quy tac chi phoi toan bo module nay:

1. Parse phong thu. Gan nhu moi the deu optional va nhieu the lap lai.
   Khong bao gio gia dinh mot the co mat.

2. Normalize dung thu tu. Text block bi wrap cung bang `&#xD;` (CR). Xoa CR
   la dung, nhung `' '.join(text.split())` thi SAI: no xoa luon ky tu xuong
   dong dang phan tach cac bullet, va moi criteria se dinh lam mot khoi.
   Loi nay khong bao ra ngoai — no chi lam so criteria tut tu 13.3 xuong 1.0.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Dau dong dang bullet. Co tinh KHONG nhan dang `a.` / `i.` (chu cai + dau cham)
# vi dong tiep noi bat dau bang mot chu cai roi dau cham la chuyen thuong gap,
# nhan nham se cat doi mot criterion dang hoan chinh.
BULLET_RE = re.compile(r"^(\s*)(?:[-*•‣▪·–—]|\(\d+\)|\d+[.)]|\([a-zA-Z]\)|[a-zA-Z]\))\s+")

# Dong header dung rieng mot minh: "Inclusion Criteria:", "EXCLUSION CRITERIA",
# "Key Inclusion Criteria", "Exclusion Criteria for Cohort B:"
HEADER_RE = re.compile(
    r"^\s*(?:key|main|principal|general|major)?\s*"
    r"(inclusion|exclusion|ineligibility)\b[^:\n]{0,60}?:?\s*$",
    re.I,
)

# Header co noi dung ngay tren cung dong: "Inclusion Criteria: patients must ..."
HEADER_INLINE_RE = re.compile(
    r"^\s*(?:key|main|principal|general|major)?\s*"
    r"(inclusion|exclusion|ineligibility)\b[^:\n]{0,60}?:\s*(\S.*)$",
    re.I,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9])")

_AGE_RE = re.compile(r"^\s*([\d.]+)\s*(year|month|week|day|hour|minute)s?\s*$", re.I)
_AGE_FACTOR = {
    "year": 1.0,
    "month": 1.0 / 12,
    "week": 1.0 / 52.1775,
    "day": 1.0 / 365.25,
    "hour": 1.0 / (365.25 * 24),
    "minute": 1.0 / (365.25 * 24 * 60),
}

MIN_CRITERION_CHARS = 3


@dataclass
class Criterion:
    idx: int
    section: str  # inclusion | exclusion | unknown
    text: str
    span_start: int
    span_end: int
    lead_in: str | None = None


@dataclass
class Trial:
    nct_id: str
    title: str | None = None
    summary: str | None = None
    detail: str | None = None
    gender: str | None = None
    min_age_years: float | None = None
    max_age_years: float | None = None
    min_age_raw: str | None = None
    max_age_raw: str | None = None
    healthy_volunteers: str | None = None
    phase: str | None = None
    status: str | None = None
    study_type: str | None = None
    criteria_raw: str | None = None
    parse_method: str = "none"
    conditions: list[str] = field(default_factory=list)
    interventions: list[tuple[str, str | None]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    mesh: list[tuple[str, str]] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Normalize
# --------------------------------------------------------------------------- #

def normalize_textblock(raw: str | None) -> str | None:
    """Chuan hoa mot <textblock> nhung GIU NGUYEN ranh gioi dong.

    Chi lam ba viec: thong nhat ky tu xuong dong, cat khoang trang cuoi dong,
    va bo phan thut le chung. Khong gop dong, vi dong chinh la thu phan tach
    cac criteria.
    """
    if raw is None:
        return None
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if indents:
        d = min(indents)
        lines = [ln[d:] if len(ln) >= d else ln.lstrip() for ln in lines]
    out = "\n".join(lines).strip("\n")
    return out or None


def flatten(raw: str | None) -> str | None:
    """Gop thanh mot dong. Chi dung cho truong KHONG can offset (title, summary)."""
    if raw is None:
        return None
    s = " ".join(raw.replace("\r", " ").split())
    return s or None


def parse_age(raw: str | None) -> tuple[float | None, str | None]:
    """'14 Years' -> (14.0, '14 Years');  'N/A' -> (None, 'N/A').

    Tra ve None khi khong xac dinh duoc. Nguoi goi PHAI coi None la 'khong
    biet', khong duoc thay bang 0 hay vo cuc — do la cho invariant 2 bi pha
    lan dau tien va am tham nhat.
    """
    if raw is None:
        return None, None
    raw = raw.strip()
    if not raw:
        return None, None
    m = _AGE_RE.match(raw)
    if not m:
        return None, raw  # 'N/A' va cac dang la khac
    return float(m.group(1)) * _AGE_FACTOR[m.group(2).lower()], raw


# --------------------------------------------------------------------------- #
# Tach criteria
# --------------------------------------------------------------------------- #

def _locate(blob: str, lo: int, hi: int, text: str) -> tuple[int, int]:
    """Tim vi tri that cua `text` trong blob[lo:hi].

    Text da bi gop khoang trang con blob thi con nguyen xuong dong, nen phai
    khop mem: moi khoang trang trong text co the ung voi bat ky chuoi khoang
    trang nao trong blob. Khong tim thay thi lui ve span cua ca block.
    """
    tokens = text.split()
    if not tokens:
        return lo, hi
    pattern = r"\s+".join(re.escape(tok) for tok in tokens)
    m = re.search(pattern, blob[lo:hi])
    return (lo + m.start(), lo + m.end()) if m else (lo, hi)


def _classify(line: str) -> str:
    if not line.strip():
        return "blank"
    if HEADER_RE.match(line) or HEADER_INLINE_RE.match(line):
        return "header"
    if BULLET_RE.match(line):
        return "bullet"
    return "text"


def _section_of(line: str) -> str:
    m = HEADER_RE.match(line) or HEADER_INLINE_RE.match(line)
    kind = m.group(1).lower()
    return "inclusion" if kind == "inclusion" else "exclusion"


def segment_criteria(blob: str | None) -> tuple[list[Criterion], str]:
    """Tach blob eligibility thanh tung criterion rieng le.

    Tra ve (danh sach criterion, phuong phap da dung). Moi criterion mang
    span tro nguoc vao chinh `blob` da normalize, nen co the kiem chung
    `blob[span_start:span_end]` khop voi text — day la co so ky thuat cua
    invariant 3 (moi ket luan phai truy nguoc ve nguon).
    """
    if not blob:
        return [], "none"

    # Gom thanh cac "block": moi block la 1 bullet + cac dong tiep noi cua no,
    # hoac 1 doan van tho khi khong co bullet.
    section = "unknown"
    lead_in: str | None = None
    blocks: list[dict] = []
    cur: dict | None = None
    saw_bullet = False
    saw_number = False

    offset = 0
    for line in blob.split("\n"):
        start = offset
        offset += len(line) + 1  # +1 cho '\n'
        kind = _classify(line)

        if kind == "blank":
            continue

        if kind == "header":
            cur = None
            lead_in = None
            section = _section_of(line)
            m = HEADER_INLINE_RE.match(line)
            if m:  # con noi dung sau dau hai cham -> xu ly nhu dong text
                s2 = start + m.start(2)
                cur = {"section": section, "start": s2, "end": start + len(line),
                       "parts": [m.group(2).strip()], "lead_in": None}
                blocks.append(cur)
            continue

        if kind == "bullet":
            m = BULLET_RE.match(line)
            body = line[m.end():].strip()

            # Header nam duoi dang bullet: "-  Exclusion Criteria:". Neu coi no
            # la mot criterion thi section khong chuyen, va MOI criterion sau do
            # bi gan nhan sai. O Phase 8 loi nay lat nguoc ket luan: mot tieu chi
            # loai tru bi coi la tieu chi thu nhan se doi violated thanh satisfied.
            if HEADER_RE.match(body) or HEADER_INLINE_RE.match(body):
                cur = None
                lead_in = None
                section = _section_of(body)
                mi = HEADER_INLINE_RE.match(body)
                if mi:  # con noi dung sau dau hai cham
                    s2 = start + m.end() + mi.start(2)
                    cur = {"section": section, "start": s2, "end": start + len(line),
                           "parts": [mi.group(2).strip()], "lead_in": None}
                    blocks.append(cur)
                continue

            saw_bullet = True
            if re.match(r"^\s*(?:\(\d+\)|\d+[.)])", line):
                saw_number = True
            s2 = start + m.end()
            cur = {"section": section, "start": s2, "end": start + len(line),
                   "parts": [body], "lead_in": lead_in}
            blocks.append(cur)
            continue

        # kind == "text"
        stripped = line.strip()
        if cur is not None:
            # dong tiep noi cua criterion dang mo
            cur["parts"].append(stripped)
            cur["end"] = start + len(line)
        elif stripped.endswith(":"):
            # cau dan truoc mot danh sach, vd "Acute onset of:"
            lead_in = stripped
        else:
            # doan van tho, chua thay bullet nao
            cur = {"section": section, "start": start, "end": start + len(line),
                   "parts": [stripped], "lead_in": lead_in}
            blocks.append(cur)

    # Neu tuyet nhien khong co bullet nao -> cat theo cau.
    if not saw_bullet:
        out: list[Criterion] = []
        for b in blocks:
            text = " ".join(b["parts"]).strip()
            for piece in SENTENCE_SPLIT_RE.split(text):
                piece = piece.strip()
                if len(piece) < MIN_CRITERION_CHARS:
                    continue
                # Cat theo cau thi span cua block khong con dung cho tung cau,
                # nen phai do lai vi tri that cua cau trong blob.
                s, e = _locate(blob, b["start"], b["end"], piece)
                out.append(Criterion(len(out), b["section"], piece, s, e, b["lead_in"]))
        return out, ("sentence_split" if out else "none")

    out = []
    for b in blocks:
        text = " ".join(p for p in b["parts"] if p).strip()
        if len(text) < MIN_CRITERION_CHARS:
            continue
        out.append(Criterion(len(out), b["section"], text,
                             b["start"], b["end"], b["lead_in"]))

    method = "mixed" if (saw_number and saw_bullet and
                         re.search(r"^\s*[-*•]", blob, re.M)) else \
             "numbered" if saw_number else "bulleted"
    return out, method


# --------------------------------------------------------------------------- #
# XML -> Trial
# --------------------------------------------------------------------------- #

def _text(root: ET.Element, path: str) -> str | None:
    el = root.find(path)
    return el.text if el is not None else None


def parse_trial(path: str) -> Trial | None:
    """Doc mot file XML. Tra ve None neu file hong hoac thieu nct_id."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    nct_id = flatten(_text(root, "id_info/nct_id"))
    if not nct_id:
        return None

    criteria_raw = normalize_textblock(_text(root, "eligibility/criteria/textblock"))
    criteria, method = segment_criteria(criteria_raw)

    min_y, min_raw = parse_age(_text(root, "eligibility/minimum_age"))
    max_y, max_raw = parse_age(_text(root, "eligibility/maximum_age"))

    t = Trial(
        nct_id=nct_id,
        title=flatten(_text(root, "brief_title")),
        summary=flatten(_text(root, "brief_summary/textblock")),
        detail=flatten(_text(root, "detailed_description/textblock")),
        gender=flatten(_text(root, "eligibility/gender")),
        min_age_years=min_y,
        max_age_years=max_y,
        min_age_raw=min_raw,
        max_age_raw=max_raw,
        healthy_volunteers=flatten(_text(root, "eligibility/healthy_volunteers")),
        phase=flatten(_text(root, "phase")),
        status=flatten(_text(root, "overall_status")),
        study_type=flatten(_text(root, "study_type")),
        criteria_raw=criteria_raw,
        parse_method=method,
        criteria=criteria,
    )

    t.conditions = [flatten(e.text) for e in root.findall("condition")]
    t.conditions = [c for c in t.conditions if c]
    t.keywords = [flatten(e.text) for e in root.findall("keyword")]
    t.keywords = [k for k in t.keywords if k]

    for iv in root.findall("intervention"):
        name = flatten(_text(iv, "intervention_name"))
        if name:
            t.interventions.append((name, flatten(_text(iv, "intervention_type"))))

    for src in ("condition_browse", "intervention_browse"):
        for e in root.findall(f"{src}/mesh_term"):
            term = flatten(e.text)
            if term:
                t.mesh.append((term, src.replace("_browse", "")))

    return t
