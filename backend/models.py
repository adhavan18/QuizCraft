from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]
Generator = Literal["auto", "gemini", "offline"]


class Question(BaseModel):
    question: str
    options: list[str]
    answer: str = Field(description="Must exactly match one of the options")
    explanation: str = ""


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: Difficulty = "medium"
    # auto = Gemini when GOOGLE_API_KEY is set, otherwise the offline NLP pipeline
    generator: Generator = "auto"


class QuizResponse(BaseModel):
    topic: str
    difficulty: Difficulty
    generator: str
    questions: list[Question]
    sources: list[str] = []
