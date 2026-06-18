import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def get_llm(temperature: float = 0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in your .env file.")
    
    return LLM(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=temperature,
    )