"""
Extract speaker name from BBTC sermon filename conventions.

Internal helper used by filename_parser.py.
"""

import re
from src.storage.normalize_speaker import normalize_speaker

_ABBREVS = {
    "SPDanFoo":     "SP Daniel Foo",
    "SPDanielFoo":  "SP Daniel Foo",
    "DF":           "SP Daniel Foo",
    "DanFoo":       "SP Daniel Foo",
    "SP_Dan_Foo":   "SP Daniel Foo",
    "PSL":          "SP Chua Seng Lee",
    "pCSL":         "SP Chua Seng Lee",
    "CSL":          "SP Chua Seng Lee",
    "ChuaSengLee":  "SP Chua Seng Lee",
    "eLVM":         "Elder Lok Vi Ming",
    "eLKG":         "Ps Low Kok Guan",
    "eGHC":         "Elder Goh Hock Chye",
    "eHTL":         "Elder Ho Tuck Leh",
    "eES":          "Ps Edric Sng",
    "pES":          "Ps Edric Sng",
    "JG":           "Jeffrey Goh",
}

_CAMEL_RE = re.compile(
    r'(?<![A-Za-z])(SP|DSP|Ps|Rev(?:Dr)?|Dr|Elder|CS|e)'
    r'([A-Z][a-z]+(?:[A-Z][a-z]+)+)',
)


def speaker_from_filename(filename: str) -> str | None:
    """Return normalized speaker name extracted from filename, or None."""
    stem = re.sub(r'\.(pdf|pptx?|docx?)$', '', filename, flags=re.IGNORECASE)
    stem = re.sub(r'^(English|Mandarin)_\d{4}_', '', stem)

    for seg in re.split(r'[_\-]', stem):
        if seg in _ABBREVS:
            return _ABBREVS[seg]

    m = _CAMEL_RE.search(stem)
    if m:
        prefix, body = m.group(1), m.group(2)
        if prefix == 'e':
            prefix = 'Elder'
        elif prefix == 'CS':
            prefix = 'Ps'
        words = re.findall(r'[A-Z][a-z]+', body)
        if words:
            candidate = f"{prefix} {' '.join(words)}"
            result = normalize_speaker(candidate)
            if result:
                return result

    m = re.search(
        r'\bby[-\s]+((?:SP|DSP|Ps|Pastor|Elder|Dr|Rev)\s+)?'
        r'([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+)',
        stem.replace('-', ' '),
        re.IGNORECASE,
    )
    if m:
        title = (m.group(1) or '').strip()
        name = m.group(2).strip()
        candidate = f"{title} {name}".strip()
        result = normalize_speaker(candidate)
        if result:
            return result

    return None
