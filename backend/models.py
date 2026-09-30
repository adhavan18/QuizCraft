from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]


class Question(BaseModel):
    question: str
    options: list[str]
    answer: str = Field(description="Must exactly match one of the options")
    explanation: str = ""


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: Difficulty = "medium"


class QuizResponse(BaseModel):
    topic: str
    difficulty: Difficulty
    questions: list[Question]
    sources: list[str] = []
