"""
One-time backfill: re-extract speaker, date, topic from filenames for all
existing records in data/sermons.db.

Run from project root:
    python backfill_metadata.py [--dry-run]
"""

import sys
import sqlite3
from src.ingestion.file_classifier import classify_file
from src.ingestion.filename_parser import parse_cell_guide_filename
from src.ingestion.speaker_from_filename import speaker_from_filename
from src.storage.sqlite_store import SermonRegistry

DB_PATH = "data/sermons.db"
DRY_RUN = "--dry-run" in sys.argv


def main():
    # Ensure topic column exists
    SermonRegistry(DB_PATH)

    changed = 0
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT sermon_id, filename, speaker, date, topic FROM sermons"
        ).fetchall()

        for sermon_id, filename, cur_speaker, cur_date, cur_topic in rows:
            kind = classify_file(filename)
            updates: dict = {}

            try:
                if kind == "cell_guide":
                    parsed = parse_cell_guide_filename(filename)
                    if parsed.get("speaker") and parsed["speaker"] != cur_speaker:
                        updates["speaker"] = parsed["speaker"]
                    if parsed.get("date") and parsed["date"] != cur_date:
                        updates["date"] = parsed["date"]
                    if parsed.get("topic") and parsed["topic"] != cur_topic:
                        updates["topic"] = parsed["topic"]

                elif kind == "sermon_slides":
                    new_sp = speaker_from_filename(filename)
                    if new_sp and new_sp != cur_speaker:
                        updates["speaker"] = new_sp
            except Exception as e:
                print(f"  ERROR parsing {filename}: {e}")
                continue

            if updates:
                changed += 1
                label = f"[{kind}] {filename}"
                for k, v in updates.items():
                    old_val = cur_speaker if k == "speaker" else (cur_date if k == "date" else cur_topic)
                    print(f"  {k}: {old_val!r} → {v!r}  ({label})")
                if not DRY_RUN:
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    conn.execute(
                        f"UPDATE sermons SET {set_clause} WHERE sermon_id = ?",
                        (*updates.values(), sermon_id),
                    )

        if not DRY_RUN:
            conn.commit()

    mode = "DRY RUN — " if DRY_RUN else ""
    print(f"\n{mode}{changed} record(s) updated.")


if __name__ == "__main__":
    main()
