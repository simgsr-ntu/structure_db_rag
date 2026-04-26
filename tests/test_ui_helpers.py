import sqlite3
import pytest
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar


# ── extract_chart_path ────────────────────────────────────────────────────────

class TestExtractChartPath:
    def test_detects_path_in_response(self):
        response = "Here is the chart: /tmp/bbtc_chart_abc12345.png"
        _, path = extract_chart_path(response)
        assert path == "/tmp/bbtc_chart_abc12345.png"

    def test_strips_path_from_text(self):
        response = "Here is the chart: /tmp/bbtc_chart_abc12345.png"
        text, _ = extract_chart_path(response)
        assert "/tmp/bbtc_chart" not in text

    def test_strips_trailing_colon_artifact(self):
        response = "Here is the chart: /tmp/bbtc_chart_abc12345.png"
        text, _ = extract_chart_path(response)
        assert text == "Here is the chart"

    def test_returns_none_when_no_path(self):
        response = "No chart here, just text."
        text, path = extract_chart_path(response)
        assert path is None
        assert text == "No chart here, just text."

    def test_default_label_when_only_path(self):
        response = "/tmp/bbtc_chart_abc12345.png"
        text, path = extract_chart_path(response)
        assert path == "/tmp/bbtc_chart_abc12345.png"
        assert text == "Here is the chart:"

    def test_non_hex_filename_not_matched(self):
        # 'x' is not in [a-f0-9] — should not match
        response = "See /tmp/bbtc_chart_xyz99999.png for details"
        _, path = extract_chart_path(response)
        assert path is None

    def test_preserves_text_before_and_after_path(self):
        response = "Preface text. /tmp/bbtc_chart_aabb1234.png More text."
        text, path = extract_chart_path(response)
        assert path == "/tmp/bbtc_chart_aabb1234.png"
        assert "Preface text." in text
        assert "More text." in text


# ── fetch_archive_stats ───────────────────────────────────────────────────────

class TestFetchArchiveStats:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = str(tmp_path / "test.db")
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE sermons (
                    sermon_id TEXT PRIMARY KEY,
                    speaker TEXT,
                    year INTEGER,
                    language TEXT
                )
            """)
            conn.executemany(
                "INSERT INTO sermons VALUES (?, ?, ?, ?)",
                [
                    ("s1", "Pastor A", 2022, "English"),
                    ("s2", "Pastor A", 2023, "English"),
                    ("s3", "Pastor B", 2024, "Mandarin"),
                    ("s4", None, None, None),
                ],
            )
        return path

    def test_sermon_count_includes_all_rows(self, db_path):
        assert fetch_archive_stats(db_path)["sermons"] == 4

    def test_speaker_count_excludes_null(self, db_path):
        assert fetch_archive_stats(db_path)["speakers"] == 2

    def test_speaker_count_excludes_empty_string(self, db_path):
        # Add a row with empty-string speaker
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO sermons VALUES (?, ?, ?, ?)", ("s5", "", 2023, "English"))
        assert fetch_archive_stats(db_path)["speakers"] == 2  # still 2, not 3

    def test_year_range(self, db_path):
        stats = fetch_archive_stats(db_path)
        assert stats["year_min"] == 2022
        assert stats["year_max"] == 2024

    def test_language_count_excludes_null(self, db_path):
        assert fetch_archive_stats(db_path)["languages"] == 2

    def test_returns_none_when_db_missing(self):
        assert fetch_archive_stats("/nonexistent/db.sqlite") is None


# ── render_stats_bar ──────────────────────────────────────────────────────────

class TestRenderStatsBar:
    def test_renders_all_stat_fields(self):
        stats = {
            "sermons": 847, "speakers": 14,
            "year_min": 2018, "year_max": 2024, "languages": 2,
        }
        html = render_stats_bar(stats)
        assert "847 sermons" in html
        assert "14 speakers" in html
        assert "2018" in html
        assert "2024" in html
        assert "2 languages" in html

    def test_fallback_html_when_stats_none(self):
        html = render_stats_bar(None)
        assert "unavailable" in html.lower()
        assert "stats-bar" in html

    def test_renders_na_when_year_is_none(self):
        stats = {
            "sermons": 10, "speakers": 3,
            "year_min": None, "year_max": None, "languages": 1,
        }
        html = render_stats_bar(stats)
        assert "N/A" in html
        assert "10 sermons" in html

    def test_renders_na_when_only_year_max_is_none(self):
        stats = {
            "sermons": 10, "speakers": 3,
            "year_min": 2020, "year_max": None, "languages": 1,
        }
        html = render_stats_bar(stats)
        assert "N/A" in html
        assert "None" not in html
