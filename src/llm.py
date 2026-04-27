from langchain_ollama import ChatOllama


def get_llm(temperature=0, ollama_model="llama3.1:8b"):
    return ChatOllama(model=ollama_model, temperature=temperature)
