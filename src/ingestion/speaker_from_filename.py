"""
Extract speaker name from BBTC sermon filename conventions.

BBTC filenames embed the speaker in two ways:
  1. Short abbreviations:  _eLKG_, _PSL_, _SPDanFoo_, _DF_
  2. CamelCase full name:  _PsAndrewTan_, _RevRickSeaward_, _eLVM_

This is tried BEFORE the LLM so garbled PDF text doesn't matter.
"""

import re
from src.storage.normalize_speaker import normalize_speaker

# Abbreviations that appear as isolated underscore/hyphen segments in filenames
_ABBREVS = {
    # SP Daniel Foo
    "SPDanFoo":     "SP Daniel Foo",
    "SPDanielFoo":  "SP Daniel Foo",
    "DF":           "SP Daniel Foo",
    # SP Chua Seng Lee  (PSL = Ps Seng Lee; pCSL = Ps Chua Seng Lee)
    "PSL":          "SP Chua Seng Lee",
    "pCSL":         "SP Chua Seng Lee",
    "CSL":          "SP Chua Seng Lee",
    # Elders / Pastors
    "eLVM":         "Elder Lok Vi Ming",
    "eLKG":         "Ps Low Kok Guan",
    "eGHC":         "Elder Goh Hock Chye",
    "eHTL":         "Elder Ho Tuck Leh",
    # Others
    "JG":           "Jeffrey Goh",
}

# Regex: one or more title+CamelCase name glued together (no spaces)
# e.g. "PsAndrewTan", "RevRickSeaward", "SPDanFoo", "CSEdricSng"
_CAMEL_RE = re.compile(
    r'(?<![A-Za-z])(SP|DSP|Ps|Rev(?:Dr)?|Dr|Elder|CS|e)'
    r'([A-Z][a-z]+(?:[A-Z][a-z]+)+)',
)

def speaker_from_filename(filename: str) -> str | None:
    """Return normalized speaker name extracted from filename, or None."""
    # Strip extension and language/year prefix
    stem = re.sub(r'\.(pdf|pptx?|docx?)$', '', filename, flags=re.IGNORECASE)
    stem = re.sub(r'^(English|Mandarin)_\d{4}_', '', stem)

    # 1. Exact abbreviation match on each underscore/hyphen segment
    for seg in re.split(r'[_\-]', stem):
        if seg in _ABBREVS:
            return _ABBREVS[seg]

    # 2. CamelCase title+name block
    m = _CAMEL_RE.search(stem)
    if m:
        prefix, body = m.group(1), m.group(2)
        # Expand "e" prefix → "Elder"
        if prefix == 'e':
            prefix = 'Elder'
        elif prefix == 'CS':
            prefix = 'Ps'   # CS = Cell Supervisor; these are internal pastors
        # Split camelCase body into words
        words = re.findall(r'[A-Z][a-z]+', body)
        if words:
            candidate = f"{prefix} {' '.join(words)}"
            result = normalize_speaker(candidate)
            if result:
                return result

    # 3. Hyphen-delimited "by-Title-Firstname-Lastname" or "by-Firstname-Lastname"
    # e.g. "by-SP-Daniel-Foo", "by-Edric-Sng", "by-Elder-Low-Kok-Guan"
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
