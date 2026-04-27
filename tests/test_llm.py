from unittest.mock import patch, MagicMock
from langchain_ollama import ChatOllama


def test_get_llm_returns_chat_ollama():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        result = get_llm()
        mock_cls.assert_called_once()


def test_get_llm_passes_temperature():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(temperature=0.5)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("temperature") == 0.5


def test_get_llm_uses_custom_model():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(ollama_model="llama3.2:3b")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == "llama3.2:3b"
