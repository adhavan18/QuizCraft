import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from backend.models import Question

load_dotenv()  # loads .env into environment

# Older Gemini models get retired for new keys; keep the model configurable
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order when the primary model is overloaded
FALLBACK_MODELS = [m for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash").split(",") if m]
RETRYABLE = {429, 500, 503}  # rate limited / transient overload

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


def generate_with_retry(retries: int = 2, **kwargs):
    models = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    for model in models:
        for attempt in range(retries + 1):
            try:
                return get_client().models.generate_content(model=model, **kwargs)
            except errors.APIError as e:
                if e.code not in RETRYABLE:
                    raise
                if attempt == retries:
                    if model == models[-1]:
                        raise
                    break  # move on to the next model
                time.sleep(2 ** attempt * 3)  # 3, 6 s


def generate_quiz(context: str, topic: str, num_questions: int = 5, difficulty: str = "medium") -> list[Question]:
    response = generate_with_retry(
        contents=build_prompt(context, topic, num_questions, difficulty),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Question],
            temperature=0.4,
        ),
    )
    return validate(response.parsed or [])
