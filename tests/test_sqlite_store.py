import os, tempfile, sqlite3, pytest
from src.storage.sqlite_store import SermonRegistry


@pytest.fixture
def reg():
    with tempfile.TemporaryDirectory() as d:
        yield SermonRegistry(db_path=os.path.join(d, "t.db"))


def test_sermons_table_columns(reg):
    with sqlite3.connect(reg.db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sermons)").fetchall()]
    for col in ["sermon_id", "date", "year", "language", "speaker", "topic",
                "theme", "summary", "key_verse", "ng_file", "ps_file", "status"]:
        assert col in cols, f"Missing column: {col}"


def test_verses_table_columns(reg):
    with sqlite3.connect(reg.db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(verses)").fetchall()]
    for col in ["id", "sermon_id", "verse_ref", "book", "chapter",
                "verse_start", "verse_end", "is_key_verse"]:
        assert col in cols, f"Missing column: {col}"


def test_upsert_sermon(reg):
    reg.upsert_sermon({
        "sermon_id": "2024-01-06-discipleship",
        "date": "2024-01-06",
        "year": 2024,
        "language": "English",
        "speaker": "SP Chua Seng Lee",
        "topic": "The Heart of Discipleship",
        "theme": "#CanIPrayForYou",
        "summary": "A summary.",
        "key_verse": "Luke 9:23",
        "ng_file": "English_2024_06-07-Jan-2024-The-Heart-of-Discipleship-Members-Guide.pdf",
        "ps_file": None,
        "status": "indexed",
    })
    row = reg.get_sermon("2024-01-06-discipleship")
    assert row["speaker"] == "SP Chua Seng Lee"
    assert row["key_verse"] == "Luke 9:23"


def test_insert_verse(reg):
    verse = {
        "sermon_id": "2024-01-06-discipleship",
        "verse_ref": "Luke 9:23",
        "book": "Luke",
        "chapter": 9,
        "verse_start": 23,
        "verse_end": None,
        "is_key_verse": 1,
    }
    reg.upsert_sermon({
        "sermon_id": "2024-01-06-discipleship",
        "date": "2024-01-06", "year": 2024, "language": "English",
        "speaker": "SP Chua Seng Lee", "topic": "Discipleship",
        "theme": None, "summary": None, "key_verse": "Luke 9:23",
        "ng_file": "test.pdf", "ps_file": None, "status": "grouped",
    })
    reg.insert_verse(verse)
    reg.insert_verse(verse)  # duplicate — must be silently ignored
    with sqlite3.connect(reg.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM verses WHERE sermon_id = ?",
            ("2024-01-06-discipleship",)
        ).fetchone()[0]
        row = conn.execute(
            "SELECT book, is_key_verse FROM verses WHERE sermon_id = ?",
            ("2024-01-06-discipleship",)
        ).fetchone()
    assert count == 1, "Duplicate verse row was inserted"
    assert row[0] == "Luke"
    assert row[1] == 1


def test_sermon_exists(reg):
    assert not reg.sermon_exists("2024-01-06-discipleship")
    reg.upsert_sermon({
        "sermon_id": "2024-01-06-discipleship",
        "date": "2024-01-06", "year": 2024, "language": "English",
        "speaker": None, "topic": None, "theme": None,
        "summary": None, "key_verse": None,
        "ng_file": "test.pdf", "ps_file": None, "status": "grouped",
    })
    assert reg.sermon_exists("2024-01-06-discipleship")


def test_get_pending_sermons(reg):
    for sid, status in [("a", "indexed"), ("b", "grouped"), ("c", "extracted")]:
        reg.upsert_sermon({
            "sermon_id": sid, "date": "2024-01-06", "year": 2024,
            "language": "English", "speaker": None, "topic": None,
            "theme": None, "summary": None, "key_verse": None,
            "ng_file": "f.pdf", "ps_file": None, "status": status,
        })
    pending = reg.get_pending_sermons()
    ids = [s["sermon_id"] for s in pending]
    assert "b" in ids
    assert "c" in ids
    assert "a" not in ids


def test_upsert_derives_year_from_date(reg):
    reg.upsert_sermon({
        "sermon_id": "year-derive-test",
        "date": "2023-03-15",
        "language": "English",
        "speaker": None, "topic": None, "theme": None,
        "summary": None, "key_verse": None,
        "ng_file": "f.pdf", "ps_file": None, "status": "grouped",
    })
    row = reg.get_sermon("year-derive-test")
    assert row["year"] == 2023


def test_ng_file_indexed(reg):
    reg.upsert_sermon({
        "sermon_id": "ng-indexed-test",
        "date": "2024-01-06", "year": 2024, "language": "English",
        "speaker": None, "topic": None, "theme": None,
        "summary": None, "key_verse": None,
        "ng_file": "guide.pdf", "ps_file": None, "status": "grouped",
    })
    assert not reg.ng_file_indexed("guide.pdf")
    reg.mark_status("ng-indexed-test", "indexed")
    assert reg.ng_file_indexed("guide.pdf")


def test_wipe_clears_both_tables(reg):
    reg.upsert_sermon({
        "sermon_id": "wipe-test",
        "date": "2024-01-06", "year": 2024, "language": "English",
        "speaker": None, "topic": None, "theme": None,
        "summary": None, "key_verse": None,
        "ng_file": "f.pdf", "ps_file": None, "status": "indexed",
    })
    reg.insert_verse({
        "sermon_id": "wipe-test", "verse_ref": "John 3:16",
        "book": "John", "chapter": 3, "verse_start": 16,
        "verse_end": None, "is_key_verse": 1,
    })
    reg.wipe()
    assert reg.get_all_sermons() == []
    with sqlite3.connect(reg.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
    assert count == 0
