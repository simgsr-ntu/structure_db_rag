# src/ingestion/metadata_extractor.py
import json
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm

load_dotenv()

_SYSTEM = """You are a metadata extraction assistant for church sermon files.
Given the first 500 characters of a sermon document, extract:
- speaker: the pastor or preacher's name (string or null)
- date: the sermon date as YYYY-MM-DD (string or null)
- series: the sermon series name (string or null)
- bible_book: the primary Bible book referenced (string or null)
- primary_verse: the key verse e.g. "Romans 8:28" (string or null)

Respond ONLY with a valid JSON object. No explanation. No markdown fences."""

_EMPTY = {"speaker": None, "date": None, "series": None, "bible_book": None, "primary_verse": None}

_OLLAMA_FALLBACK_MODEL = "llama3.2:3b"

_RATE_LIMIT_MARKERS = ("rate_limit", "429", "quota", "tokens per day", "tpd")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


class MetadataExtractor:
    def __init__(self):
        self._groq = get_llm(temperature=0, groq_model="llama-3.1-8b-instant")
        self._ollama: ChatOllama | None = None  # lazy-init only when needed

    def _get_ollama(self) -> ChatOllama:
        if self._ollama is None:
            self._ollama = ChatOllama(model=_OLLAMA_FALLBACK_MODEL, temperature=0)
        return self._ollama

    def _invoke(self, messages: list) -> str:
        try:
            return self._groq.invoke(messages).content.strip()
        except Exception as e:
            if _is_rate_limit(e):
                print(f"⚠️  Groq rate limit — falling back to Ollama ({_OLLAMA_FALLBACK_MODEL})")
                return self._get_ollama().invoke(messages).content.strip()
            raise

    def extract(self, text_preview: str) -> dict:
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=text_preview[:500]),
        ]
        try:
            raw = self._invoke(messages)
            return json.loads(raw)
        except Exception:
            return _EMPTY.copy()
