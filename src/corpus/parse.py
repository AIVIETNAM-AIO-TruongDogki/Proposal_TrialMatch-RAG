"""Parse ClinicalTrials.gov XML into a trial record + segmented criteria.

Two rules govern this whole module:

1. Parse defensively. Almost every tag is optional and many repeat. Never
   assume a tag is present.

2. Normalize in the right order. Text blocks are wrapped with literal
   `&#xD;` (CR). Stripping CR is correct, but `' '.join(text.split())` is
   NOT — it also erases the newlines that separate bullets, collapsing every
   criterion into one block. This fails silently: it just drops the average
   criteria count from 13.3 to 1.0.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Bullet-line markers. Deliberately does NOT match `a.` / `i.` (letter + dot)
# — a continuation line starting with a letter then a period is common, and
# misdetecting it would split one criterion in half.
BULLET_RE = re.compile(r"^(\s*)(?:[-*•‣▪·–—]|\(\d+\)|\d+[.)]|\([a-zA-Z]\)|[a-zA-Z]\))\s+")

# A header line standing alone: "Inclusion Criteria:", "EXCLUSION CRITERIA",
# "Key Inclusion Criteria", "Exclusion Criteria for Cohort B:"
HEADER_RE = re.compile(
    r"^\s*(?:key|main|principal|general|major)?\s*"
    r"(inclusion|exclusion|ineligibility)\b[^:\n]{0,60}?:?\s*$",
    re.I,
)

# A header with content on the same line: "Inclusion Criteria: patients must ..."
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
    """Normalize a <textblock> while KEEPING line boundaries intact.

    Only three things: unify line endings, trim trailing whitespace per line,
    and strip common indentation. Never joins lines — lines are what
    separates criteria.
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
    """Collapse to one line. Only for fields that need no offset (title, summary)."""
    if raw is None:
        return None
    s = " ".join(raw.replace("\r", " ").split())
    return s or None


def parse_age(raw: str | None) -> tuple[float | None, str | None]:
    """'14 Years' -> (14.0, '14 Years');  'N/A' -> (None, 'N/A').

    Returns None when undetermined. Callers MUST treat None as "unknown",
    never substitute 0 or infinity — this is the first and most silent place
    invariant 2 gets broken.
    """
    if raw is None:
        return None, None
    raw = raw.strip()
    if not raw:
        return None, None
    m = _AGE_RE.match(raw)
    if not m:
        return None, raw  # 'N/A' and other unparseable forms
    return float(m.group(1)) * _AGE_FACTOR[m.group(2).lower()], raw


# --------------------------------------------------------------------------- #
# Tach criteria
# --------------------------------------------------------------------------- #

def _locate(blob: str, lo: int, hi: int, text: str) -> tuple[int, int]:
    """Find `text`'s real position within blob[lo:hi].

    `text` has had whitespace collapsed while `blob` still has newlines, so
    matching must be fuzzy: any whitespace in `text` can match any run of
    whitespace in `blob`. Falls back to the whole block's span if not found.
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
    """Split the eligibility blob into individual criteria.

    Returns (criteria list, method used). Each criterion carries a span back
    into the normalized `blob`, so `blob[span_start:span_end]` can be checked
    against its text — this is the technical basis of invariant 3 (every
    conclusion must trace back to its source).
    """
    if not blob:
        return [], "none"

    # Group into "blocks": one bullet + its continuation lines, or one raw
    # paragraph when there's no bullet at all.
    section = "unknown"
    lead_in: str | None = None
    blocks: list[dict] = []
    cur: dict | None = None
    saw_bullet = False
    saw_number = False

    offset = 0
    for line in blob.split("\n"):
        start = offset
        offset += len(line) + 1  # +1 for '\n'
        kind = _classify(line)

        if kind == "blank":
            continue

        if kind == "header":
            cur = None
            lead_in = None
            section = _section_of(line)
            m = HEADER_INLINE_RE.match(line)
            if m:  # content after the colon -> treat as a text line
                s2 = start + m.start(2)
                cur = {"section": section, "start": s2, "end": start + len(line),
                       "parts": [m.group(2).strip()], "lead_in": None}
                blocks.append(cur)
            continue

        if kind == "bullet":
            m = BULLET_RE.match(line)
            body = line[m.end():].strip()

            # A header disguised as a bullet: "-  Exclusion Criteria:". Treating
            # it as a criterion would leave `section` unchanged, mislabeling
            # every criterion after it — in Phase 8 this flips a conclusion:
            # an exclusion criterion read as inclusion turns `violated` into
            # `satisfied`.
            if HEADER_RE.match(body) or HEADER_INLINE_RE.match(body):
                cur = None
                lead_in = None
                section = _section_of(body)
                mi = HEADER_INLINE_RE.match(body)
                if mi:  # content after the colon
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
            # continuation line of the open criterion
            cur["parts"].append(stripped)
            cur["end"] = start + len(line)
        elif stripped.endswith(":"):
            # a lead-in before a list, e.g. "Acute onset of:"
            lead_in = stripped
        else:
            # raw paragraph, no bullet seen yet
            cur = {"section": section, "start": start, "end": start + len(line),
                   "parts": [stripped], "lead_in": lead_in}
            blocks.append(cur)

    # If there's no bullet at all -> split by sentence.
    if not saw_bullet:
        out: list[Criterion] = []
        for b in blocks:
            text = " ".join(b["parts"]).strip()
            for piece in SENTENCE_SPLIT_RE.split(text):
                piece = piece.strip()
                if len(piece) < MIN_CRITERION_CHARS:
                    continue
                # Splitting by sentence invalidates the block's span for each
                # piece, so re-locate each sentence's real position in the blob.
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
    """Read one XML file. Returns None if malformed or missing nct_id."""
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
