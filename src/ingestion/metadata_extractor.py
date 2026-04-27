import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

_SYSTEM = """You are a metadata extraction assistant for church sermon files.
Given the first 500 characters of a sermon document, extract:
- speaker: the pastor or preacher's name (string or null)
- date: the sermon date as YYYY-MM-DD (string or null)
- series: the sermon series name (string or null)
- bible_book: the primary Bible book referenced (string or null)
- primary_verse: the key verse e.g. "Romans 8:28" (string or null)

Respond ONLY with a valid JSON object. No explanation. No markdown fences."""

_EMPTY = {"speaker": None, "date": None, "series": None, "bible_book": None, "primary_verse": None}


class MetadataExtractor:
    def __init__(self):
        self._llm = ChatOllama(model="llama3.2:3b", temperature=0)

    def extract(self, text_preview: str) -> dict:
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=text_preview[:500]),
        ]
        try:
            raw = self._llm.invoke(messages).content.strip()
            return json.loads(raw)
        except Exception:
            return _EMPTY.copy()
