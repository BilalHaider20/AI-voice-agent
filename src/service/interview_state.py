"""Interview state schema for the LangGraph state machine."""

from typing import TypedDict, List, Optional


class QuestionRecord(TypedDict):
    question_id: str
    question_number: int
    question: str
    answer: str
    candidate_answer: str
    score: Optional[int]
    feedback: Optional[str]


class InterviewState(TypedDict):
    """State for the interview agent."""
    
    # Interview context
    # job_title: str
    # experience_level: str
    category: str
    skill: str
    
    # Interview progress
    current_question_index: int
    questions_asked: List[QuestionRecord]
    
    # Interview results
    total_score: Optional[int]
    overall_feedback: Optional[str]
    report: Optional[dict]
    
    
    # Interview status
    is_complete: bool # True if interview is completed
    current_stage: str # "greeting", "question", "feedback", "completed"
    
    # Interview configuration
    max_questions: int # Maximum number of questions to ask
    # time_limit: Optional[int]
    
    # Interview session
    # session_id: str
    # room_id: str
