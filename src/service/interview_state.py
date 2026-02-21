from typing import Annotated, List, Optional, TypedDict, Union, Dict
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class QuestionRecord(TypedDict):
    id: str
    question: str
    user_answer: Optional[str]
    # score: int
    # feedback: Dict[str, str]


# --- STATE MANAGEMENT ---
class InterviewState(TypedDict):
    """
    Maintains the state of the interview, including history and progress.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    category: str | None
    skill: str
    max_questions: int
    # questions_asked is the single source of truth for question tracking.
    # Use len(questions_asked) instead of a separate question_count.
    questions_asked: List[QuestionRecord]
    # Flags to control flow
    is_off_topic: bool
    is_complete: bool



# Validation
class ValidationResult(BaseModel):
    """Whether the candidate's response is off-topic."""
    is_off_topic: bool = Field(
        description="True if the response is off-topic or casual chat, "
                    "False if it's a genuine attempt to answer the question."
        )    