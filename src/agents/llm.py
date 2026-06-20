import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def get_llm(temperature: float = 0.2):
    api_key = os.getenv("GEMINI_API_KEY")

    # Fallback to Streamlit secrets (for Streamlit Cloud deployment,
    # where secrets are not written into a .env file)
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in your .env file (local) "
            "or in Streamlit Cloud Secrets (deployment)."
        )

    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=temperature,
    )