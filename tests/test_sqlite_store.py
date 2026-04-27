import os
import tempfile
import sqlite3
import pytest
from src.storage.sqlite_store import SermonRegistry


def test_topic_column_exists():
    with tempfile.TemporaryDirectory() as d:
        reg = SermonRegistry(db_path=os.path.join(d, "t.db"))
        with sqlite3.connect(reg.db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(sermons)").fetchall()]
        assert "topic" in cols


def test_insert_sermon_stores_topic():
    with tempfile.TemporaryDirectory() as d:
        reg = SermonRegistry(db_path=os.path.join(d, "t.db"))
        reg.insert_sermon({
            "sermon_id": "test-001",
            "filename": "test.pdf",
            "url": "http://example.com/test.pdf",
            "speaker": "SP Daniel Foo",
            "date": "2018-07-28",
            "topic": "Know Your Enemy",
            "status": "indexed",
        })
        with sqlite3.connect(reg.db_path) as conn:
            row = conn.execute(
                "SELECT topic FROM sermons WHERE sermon_id = 'test-001'"
            ).fetchone()
        assert row[0] == "Know Your Enemy"
