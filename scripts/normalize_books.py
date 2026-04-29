#!/usr/bin/env python3
"""One-time migration: normalize book names in the verses table.

Usage:
    python scripts/normalize_books.py [--db data/sermons.db] [--dry-run]
"""
import argparse
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.normalize_book import normalize_book

# Ambiguous unnumbered books that normalize_book() cannot resolve alone.
# key → (book1, book2, max_ch_book1, max_ch_book2)
# book1 is the default when the chapter is ambiguous.
_AMBIGUOUS: dict[str, tuple] = {
    "samuel":      ("1 Samuel",      "2 Samuel",      31, 24),
    "kings":       ("1 Kings",       "2 Kings",       22, 25),
    "chronicles":  ("1 Chronicles",  "2 Chronicles",  29, 36),
    "corinthians": ("1 Corinthians", "2 Corinthians", 16, 13),
    "timothy":     ("1 Timothy",     "2 Timothy",     6,  4),
    "peter":       ("1 Peter",       "2 Peter",       5,  3),
}


def _disambiguate(key: str, chapter) -> str:
    book1, book2, max1, max2 = _AMBIGUOUS[key]
    if chapter is None:
        return book1
    ch = int(chapter)
    if ch > max1 and ch > max2:
        return book1   # invalid chapter — use default
    if ch > max1:
        return book2   # exceeds book1's max → must be book2
    if ch > max2:
        return book1   # exceeds book2's max → must be book1
    return book1       # ambiguous overlap — default to book1


def _build_verse_ref(book: str, chapter, verse_start, verse_end) -> str:
    if chapter is None:
        return book
    if verse_start is None:
        return f"{book} {chapter}"
    if verse_end is not None:
        return f"{book} {chapter}:{verse_start}-{verse_end}"
    return f"{book} {chapter}:{verse_start}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/sermons.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without modifying the DB")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, sermon_id, verse_ref, book, chapter, "
        "verse_start, verse_end, is_key_verse FROM verses"
    ).fetchall()
    conn.row_factory = None

    updates: dict[int, tuple[str, str]] = {}   # row_id → (new_book, new_verse_ref)
    garbage: set[int] = set()                  # row_ids with unrecognized book
    truly_unresolved: list[str] = []

    for row in rows:
        raw = row["book"]
        chapter = row["chapter"]

        canonical = normalize_book(raw)
        if canonical is None:
            key = raw.strip().lower() if raw else ""
            if key in _AMBIGUOUS:
                canonical = _disambiguate(key, chapter)
            else:
                garbage.add(row["id"])
                truly_unresolved.append(raw or "(null)")
                continue

        new_ref = _build_verse_ref(
            canonical, row["chapter"], row["verse_start"], row["verse_end"]
        )
        if canonical != raw or new_ref != row["verse_ref"]:
            updates[row["id"]] = (canonical, new_ref)

    # Resolve post-normalization duplicates within the same sermon.
    # Two rows may now share the same (sermon_id, verse_ref) after normalization.
    # Keep the row with is_key_verse=1; otherwise keep the lowest id.
    row_by_id: dict[int, object] = {row["id"]: row for row in rows}

    def _new_key(row_id: int) -> tuple:
        if row_id in updates:
            _, new_ref = updates[row_id]
            return (row_by_id[row_id]["sermon_id"], new_ref)
        r = row_by_id[row_id]
        return (r["sermon_id"], r["verse_ref"])

    groups: dict[tuple, list[int]] = {}
    for row in rows:
        if row["id"] in garbage:
            continue
        k = _new_key(row["id"])
        groups.setdefault(k, []).append(row["id"])

    duplicates: set[int] = set()
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        ids.sort(key=lambda rid: (-(row_by_id[rid]["is_key_verse"] or 0), rid))
        for loser in ids[1:]:
            duplicates.add(loser)
            updates.pop(loser, None)

    deletes = garbage | duplicates

    if args.dry_run:
        print(f"[DRY RUN] Would update : {len(updates)} rows")
        print(f"[DRY RUN] Would delete (garbage)   : {len(garbage)} rows")
        print(f"[DRY RUN] Would delete (duplicates): {len(duplicates)} rows")
        if truly_unresolved:
            print(f"[DRY RUN] Truly unresolved books: {sorted(set(truly_unresolved))}")
        else:
            print("[DRY RUN] No unresolved books.")
        conn.close()
        return

    with conn:
        for row_id, (new_book, new_ref) in updates.items():
            conn.execute(
                "UPDATE verses SET book = ?, verse_ref = ? WHERE id = ?",
                (new_book, new_ref, row_id),
            )
        for row_id in deletes:
            conn.execute("DELETE FROM verses WHERE id = ?", (row_id,))

    conn.close()

    print(f"Updated : {len(updates)} rows")
    print(f"Deleted (garbage)   : {len(garbage)} rows")
    print(f"Deleted (duplicates): {len(duplicates)} rows")
    if truly_unresolved:
        print(f"Truly unresolved books (inspect manually): {sorted(set(truly_unresolved))}")
    else:
        print("No unresolved books.")


if __name__ == "__main__":
    main()
