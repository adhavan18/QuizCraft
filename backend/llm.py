import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.models import Question

load_dotenv()  # loads .env into environment

# gemini-1.5-flash has been retired; keep the model configurable
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DIFFICULTY_GUIDE = {
    "easy": "direct recall of facts and definitions stated in the text",
    "medium": "understanding and applying concepts from the text",
    "hard": "reasoning, comparison or multi-step application of concepts from the text",
}

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(context: str, topic: str, num_questions: int, difficulty: str) -> str:
    return f"""You are a Grade 9 NCERT science teacher writing a quiz on "{topic}".

Write exactly {num_questions} multiple-choice questions at {difficulty} difficulty
({DIFFICULTY_GUIDE[difficulty]}).

Rules:
- Use ONLY facts found in the context below; do not invent facts.
- Each question has exactly 4 distinct options and one correct answer.
- "answer" must be copied exactly from "options".
- Distractors should be plausible to a student but clearly wrong per the context.
- "explanation" is one sentence citing the relevant fact from the context.

Context:
\"\"\"{context}\"\"\"
"""


def validate(questions: list[Question]) -> list[Question]:
    # Drop malformed items rather than showing a broken question to the student
    return [q for q in questions if len(set(q.options)) == 4 and q.answer in q.options]


def generate_quiz(context: str, topic: str, num_questions: int = 5, difficulty: str = "medium") -> list[Question]:
    response = get_client().models.generate_content(
        model=MODEL_NAME,
        contents=build_prompt(context, topic, num_questions, difficulty),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Question],
            temperature=0.4,
        ),
    )
    return validate(response.parsed or [])
