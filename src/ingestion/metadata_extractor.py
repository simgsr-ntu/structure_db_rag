import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Ordered so longer book names match before shorter prefixes (e.g. "1 Corinthians" before "Corinthians")
_BIBLE_BOOKS = [
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "1 Samuel", "2 Samuel",
    "1 Corinthians", "2 Corinthians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "1 Maccabees", "2 Maccabees",
    "Song of Solomon", "Song of Songs",
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Psalm", "Proverbs", "Ecclesiastes", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
    "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "Galatians", "Ephesians", "Philippians", "Colossians", "Philemon",
    "Hebrews", "James", "Jude", "Revelation",
]
_BOOK_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in _BIBLE_BOOKS) + r")\b",
    re.IGNORECASE,
)


def _book_from_verse(verse: str | None) -> str | None:
    if not verse:
        return None
    m = _BOOK_RE.search(verse)
    return m.group(1).title() if m else None

_SYSTEM = """You are a theological metadata extraction assistant for church sermon files.
Given the text, extract:
- speaker: the preacher's full name as it appears in the text (e.g. "SP Daniel Foo", "Elder Lok Vi Ming"). Return null if not found — never return a placeholder like "Name".
- date: YYYY-MM-DD
- series: sermon series name
- bible_book: primary Bible book
- primary_verse: key verse e.g. "Romans 8:28"
- verses_used: comma-separated list of ALL other scripture references mentioned.
- summary: a 2-sentence summary of the main message.

Respond ONLY with valid JSON."""

_EMPTY = {
    "speaker": None, "date": None, "series": None, "bible_book": None, 
    "primary_verse": None, "verses_used": None, "summary": None
}


class MetadataExtractor:
    def __init__(self):
        self._llm = ChatOllama(model="llama3.2:3b", temperature=0)

    def extract(self, text_preview: str) -> dict:
        # Increase context to find speaker name if it's further down
        text_context = text_preview[:2000]
        
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Extract metadata from this sermon text preview:\n\n{text_context}"),
        ]
        try:
            res = self._llm.invoke(messages)
            raw = res.content.strip()
            # Clean potential markdown fences if the LLM ignored instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1].replace("json", "").strip()
            
            data = json.loads(raw)
            # Ensure we don't save lists or empty strings as names
            for k in data:
                if isinstance(data[k], list):
                    data[k] = ", ".join(str(x) for x in data[k])
                if data[k] == "":
                    data[k] = None
            # Derive bible_book from primary_verse or verses_used if the LLM missed it
            if not data.get("bible_book"):
                data["bible_book"] = (
                    _book_from_verse(data.get("primary_verse"))
                    or _book_from_verse(data.get("verses_used"))
                )
            return data
        except Exception as e:
            print(f"⚠️ Metadata extraction failed: {e}")
            return _EMPTY.copy()
