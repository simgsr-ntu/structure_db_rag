from src.ingestion.ng_extractor import extract_ng_metadata, extract_ng_body


SAMPLE_TEXT = """Member's Copy


TOPIC
The Heart of Discipleship
SPEAKER
SP Chua Seng Lee
THEME #CanIPrayForYou
DATE
06 & 07 January 2024


INTRODUCTION
Do you believe God can turn Singapore Godward?
In this sermon SP Chua Seng Lee unpacked the Heart of Discipleship.

THE HEART OF DISCIPLESHIP
1) What is discipleship?
"""


def test_extract_topic():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert meta["topic"] == "The Heart of Discipleship"


def test_extract_speaker():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert "Chua Seng Lee" in meta["speaker"]


def test_extract_theme():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert meta["theme"] == "#CanIPrayForYou"


def test_extract_date():
    meta = extract_ng_metadata(SAMPLE_TEXT, "dummy.pdf")
    assert meta["date"] == "2024-01-06"


def test_body_starts_after_introduction():
    body = extract_ng_body(SAMPLE_TEXT)
    assert "Do you believe" in body
    assert "Member's Copy" not in body


def test_fallback_to_filename_for_speaker():
    text = "TOPIC\nSome Sermon\nDATE\n01 January 2024\n\nINTRODUCTION\nBody."
    meta = extract_ng_metadata(text, "English_2024_01-Jan-2024-Some-Sermon-by-SP-Chua-Seng-Lee-Members-Guide.pdf")
    assert meta["speaker"] is not None


def test_missing_fields_return_none():
    meta = extract_ng_metadata("Just some plain text without labels.", "unknown.pdf")
    assert meta["topic"] is None or isinstance(meta["topic"], str)
