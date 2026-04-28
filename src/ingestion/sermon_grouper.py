"""Group BBTC sermon files into (ng, ps) sermon groups."""

from dataclasses import dataclass, field
from datetime import datetime
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import extract_any_date, extract_topic_words


@dataclass
class SermonGroup:
    ng: str | None = None
    ps: list[str] = field(default_factory=list)


def _date_proximity(d1: str | None, d2: str | None, tolerance: int = 3) -> bool:
    if not d1 or not d2:
        return False
    fmt = "%Y-%m-%d"
    try:
        return abs((datetime.strptime(d1, fmt) - datetime.strptime(d2, fmt)).days) <= tolerance
    except ValueError:
        return False


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_sermon_files(filenames: list[str]) -> list[SermonGroup]:
    """
    Group filenames into SermonGroups.
    Each NG becomes one group. PS files are paired to their NG by date proximity
    (≤ 3 days) or high topic-word Jaccard (≥ 0.5). Unpaired PS each become a
    standalone group with ng=None.
    Handout files are ignored.
    """
    ngs, pss = [], []
    for f in filenames:
        kind = classify_file(f)
        if kind == "ng":
            ngs.append(f)
        elif kind == "ps":
            pss.append(f)
        # handout: skip

    groups: list[SermonGroup] = []
    used_ps: set[str] = set()

    for ng in ngs:
        group = SermonGroup(ng=ng)
        ng_date = extract_any_date(ng)
        ng_words = extract_topic_words(ng)

        for ps in pss:
            if ps in used_ps:
                continue
            ps_date = extract_any_date(ps)
            ps_words = extract_topic_words(ps)
            if _date_proximity(ng_date, ps_date) or _jaccard(ng_words, ps_words) >= 0.5:
                group.ps.append(ps)
                used_ps.add(ps)

        groups.append(group)

    for ps in pss:
        if ps not in used_ps:
            groups.append(SermonGroup(ps=[ps]))

    return groups
