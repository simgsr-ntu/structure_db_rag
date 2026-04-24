import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(provider_type="groq", temperature=0, groq_model="llama3-70b-8192", ollama_model="llama3.1:8b", gemini_model="gemini-1.5-flash"):
    """Returns the requested LLM instance based on provider type."""
    if provider_type == "gemini" and os.getenv("GOOGLE_API_KEY"):
         print(f"🌟 Using Gemini ({gemini_model})")
         return ChatGoogleGenerativeAI(model=gemini_model, temperature=temperature)
    
    if provider_type == "groq" and os.getenv("GROQ_API_KEY"):
        print(f"📡 Using Groq ({groq_model})")
        return ChatGroq(model=groq_model, temperature=temperature)
    
    # Fallback to Ollama
    print(f"🏠 Using Ollama ({ollama_model})")
    return ChatOllama(model=ollama_model, temperature=temperature)
