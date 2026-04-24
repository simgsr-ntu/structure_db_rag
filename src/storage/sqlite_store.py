import sqlite3
import os
from datetime import datetime, timezone

class SermonRegistry:
    def __init__(self, db_path: str = "data/sermons.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sermons (
                    sermon_id TEXT PRIMARY KEY,
                    filename TEXT,
                    url TEXT UNIQUE,
                    speaker TEXT,
                    date TEXT,
                    series TEXT,
                    bible_book TEXT,
                    primary_verse TEXT,
                    language TEXT,
                    file_type TEXT,
                    year INTEGER,
                    status TEXT,
                    date_scraped TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON sermons(url)")

    def is_new(self, url: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM sermons WHERE url = ?", (url,))
            return cursor.fetchone() is None

    def insert_sermon(self, record: dict):
        cols = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        sql = f"INSERT OR REPLACE INTO sermons ({cols}) VALUES ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(record.values()))

    def mark_processed(self, url: str, status: str = "processed"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE sermons SET status = ? WHERE url = ?", (status, url))

    def get_all_sermons(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sermons")
            return [dict(row) for row in cursor.fetchall()]
