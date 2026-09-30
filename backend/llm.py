import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # loads .env into environment

# gemini-1.5-flash has been retired; keep the model configurable
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_quiz(context: str, topic: str, num_questions: int = 5) -> str:
    prompt = f"""
    Create {num_questions} multiple-choice questions (with 4 options each + correct answer)
    based on this NCERT context about {topic}:

    {context}
    """
    response = get_client().models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text
