"""Extract metadata and body text from BBTC Notes/Guide (NG) PDF text."""

import re
from src.ingestion.filename_parser import parse_cell_guide_filename
from src.storage.normalize_speaker import normalize_speaker

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date_string(raw: str) -> str | None:
    """Parse human date like '06 & 07 January 2024' → '2024-01-06'."""
    m = re.search(
        r'(\d{1,2})(?:\s*[&,\-]\s*\d{1,2})?\s+'
        r'(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
        r'\s+(\d{4})',
        raw, re.IGNORECASE,
    )
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTHS[m.group(2).lower()]
    year = int(m.group(3))
    return f"{year}-{month:02d}-{day:02d}"


def _labeled_field(text: str, label: str) -> str | None:
    """Extract value of a labeled field like 'TOPIC\\n<value>' or 'TOPIC <value>'."""
    pattern = rf'(?:^|\n)\s*{label}\s*\n\s*(.+?)(?:\n\s*[A-Z]{{3,}}|\n\s*$|$)'
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Inline: TOPIC Some Value
    m = re.search(rf'(?:^|\n)\s*{label}\s+(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_ng_metadata(text: str, filename: str) -> dict:
    """
    Extract speaker, date, topic, theme from NG text.
    Falls back to filename_parser when labeled fields are absent.
    Returns dict with keys: speaker, date, topic, theme (any may be None).
    """
    topic = _labeled_field(text, "TOPIC")
    theme = _labeled_field(text, "THEME")
    date_raw = _labeled_field(text, "DATE")
    date = _parse_date_string(date_raw) if date_raw else None

    speaker_raw = _labeled_field(text, "SPEAKER")
    speaker = normalize_speaker(speaker_raw) if speaker_raw else None

    # Fallback to filename parser for any missing field
    if not all([topic, speaker, date]):
        parsed = parse_cell_guide_filename(filename)
        topic = topic or parsed.get("topic")
        speaker = speaker or parsed.get("speaker")
        date = date or parsed.get("date")

    # Second fallback: scan filename for any date string if missing
    if not date:
        from src.ingestion.filename_parser import extract_any_date
        date = extract_any_date(filename)

    # Final fallback for year from prefix
    if not date:
        m = re.match(r'^(?:English|Mandarin)_(\d{4})_', filename)
        if m:
            date = f"{m.group(1)}-01-01"

    return {"speaker": speaker, "date": date, "topic": topic, "theme": theme}


def extract_ng_body(text: str) -> str:
    """
    Return the body of the NG: everything after the INTRODUCTION label.
    Falls back to the full text if no INTRODUCTION label found.
    """
    m = re.search(r'(?:^|\n)\s*INTRODUCTION\s*\n', text, re.IGNORECASE)
    if m:
        return text[m.end():].strip()
    # No INTRODUCTION label — strip the header block
    lines = text.split("\n")
    header_labels = {"topic", "speaker", "theme", "date", "member", "leader", "cell"}
    body_lines = []
    header_done = False
    for line in lines:
        stripped = line.strip().lower()
        if not header_done:
            if stripped in header_labels or not stripped:
                continue
            # Heuristic: header is done once we see a line > 60 chars (body prose)
            if len(line.strip()) > 60:
                header_done = True
        if header_done:
            body_lines.append(line)
    return "\n".join(body_lines).strip() or text.strip()
