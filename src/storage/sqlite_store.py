import sqlite3
import os
from datetime import datetime, timezone
from src.storage.normalize_speaker import normalize_speaker

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
                    topic TEXT,
                    language TEXT,
                    file_type TEXT,
                    year INTEGER,
                    status TEXT,
                    date_scraped TEXT
                )
            """)
            # Migrate existing databases that predate the topic column
            existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(sermons)").fetchall()]
            if "topic" not in existing_cols:
                conn.execute("ALTER TABLE sermons ADD COLUMN topic TEXT")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bible_versions (
                    version_id TEXT PRIMARY KEY,
                    filename TEXT,
                    status TEXT,
                    date_indexed TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sermon_intelligence (
                    sermon_id TEXT PRIMARY KEY,
                    speaker TEXT,
                    primary_verse TEXT,
                    verses_used TEXT,
                    summary TEXT,
                    FOREIGN KEY (sermon_id) REFERENCES sermons (sermon_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON sermons(url)")

    def is_new(self, url: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM sermons WHERE url = ?", (url,))
            return cursor.fetchone() is None

    def insert_sermon(self, record: dict):
        # Automatically normalize the speaker name before insertion
        if 'speaker' in record:
            record['speaker'] = normalize_speaker(record['speaker'])
            
        cols = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        sql = f"INSERT OR REPLACE INTO sermons ({cols}) VALUES ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(record.values()))

    def normalize_all_speakers(self):
        """Clean and consolidate all speaker names in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT speaker FROM sermons WHERE speaker IS NOT NULL")
            speakers = [row[0] for row in cursor.fetchall()]
            
            for original in speakers:
                normalized = normalize_speaker(original)
                if normalized != original:
                    cursor.execute(
                        "UPDATE sermons SET speaker = ? WHERE speaker = ?", 
                        (normalized, original)
                    )
            conn.commit()
            
    def insert_intelligence(self, intel_record: dict):
        """Insert or update sermon intelligence data."""
        cols = ", ".join(intel_record.keys())
        placeholders = ", ".join(["?"] * len(intel_record))
        sql = f"INSERT OR REPLACE INTO sermon_intelligence ({cols}) VALUES ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(intel_record.values()))

    def mark_processed(self, url: str, status: str = "processed"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE sermons SET status = ? WHERE url = ?", (status, url))

    def get_all_sermons(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sermons")
            return [dict(row) for row in cursor.fetchall()]

    def get_indexed_bibles(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM bible_versions WHERE status = 'indexed'")
            return [dict(row) for row in cursor.fetchall()]

    def mark_bible_indexed(self, version_id: str, filename: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bible_versions (version_id, filename, status, date_indexed) VALUES (?, ?, ?, ?)",
                (version_id, filename, "indexed", datetime.now().isoformat())
            )
