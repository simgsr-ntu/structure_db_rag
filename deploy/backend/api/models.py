from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    year_filter: Optional[int] = None
    speaker_filter: Optional[str] = None


class Citation(BaseModel):
    filename: str
    speaker: Optional[str] = None
    date: Optional[str] = None
    verse: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class StatsResponse(BaseModel):
    total_sermons: int
    total_speakers: int
    year_min: Optional[int] = None
    year_max: Optional[int] = None


class YearCount(BaseModel):
    year: int
    count: int


class SpeakerCount(BaseModel):
    speaker: str
    count: int


class VerseCount(BaseModel):
    bible_book: str
    count: int


class ScatterPoint(BaseModel):
    year: int
    speaker: str
    count: int
