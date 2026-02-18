"""Interview state schema for the LangGraph state machine."""

from typing import Optional, TypedDict


class QuestionRecord(TypedDict):
    """One question with ground truth and candidate answer."""

    index: int
    question: str
    answer: str  # ground truth
    answer_keywords: str
    candidate_answer: str
    score: Optional[int]
    feedback: Optional[str]


class InterviewState(TypedDict, total=False):
    """State for the interview graph. Set-once fields never change during the run."""

    # Set once at start, never change
    category: str
    skill: str
    questions: list[QuestionRecord]

    # Advances each node (only in capture_answer_node)
    current_index: int

    # Conversation
    current_question_text: str
    last_user_message: str

    # Report
    report: Optional[dict]
    interview_done: bool
    interview_started: bool
    next: str  # For routing decisions
