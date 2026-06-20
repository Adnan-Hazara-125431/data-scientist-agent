import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def get_llm(temperature: float = 0.2):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it in your .env file.")

    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=temperature,
    )