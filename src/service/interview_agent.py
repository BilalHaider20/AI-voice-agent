from livekit.agents import Agent
from .interview_state import InterviewState, QuestionRecord, ValidationResult
from .llm_service import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, Optional, TypedDict, Union, Dict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import uuid
import json
import random
import sqlite3
import pprint

load_dotenv(dotenv_path=".env.local")

llm = get_llm()

with open(os.getenv('MAP_PATH'), 'r') as f:
    map_data = json.load(f)['data']

class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI interview assistant. 
            You will ask technical interview question in one line.
            Be professional, precise and conversational. Ask one question at a time and wait for responses.
            """
        )
        
        self.interview_state = InterviewState(
            category="",
            skill="",
            max_questions=2,
            questions_asked=[],
            messages=[],
            is_off_topic=False,
            is_complete=False
        )
    
    @staticmethod
    def pre_interview_node(state: InterviewState):
        print('Interview Started')
        category = input('Please tell me which job_role you want to be interviewed on ? ')
        skill = input('Which skill? ')

        return {
            "skill": skill,
            "category": category
        }
    
    @staticmethod
    def greeting_node(state: InterviewState):
        """
        Requirement 2: Greet and set context.
        """
        category = state["category"]
        skill = state["skill"]

        greeting = (
            f"Hello! I am your AI Interviewer. Today we will be conducting an interview "
            f"focused on **{category}** specifically regarding **{skill}** skills. "
            f"Let's begin."
        )
        return {"messages": [AIMessage(content=greeting)]}
    
    @staticmethod
    def retrieve_question_id(state: InterviewState) -> Optional[str]:
        """
        Fetch a random, un-asked question ID from map.json.
        Returns None if category/skill not found or all questions exhausted.
        """
        category = state.get("category")
        skill = state.get("skill")
        difficulty = "easy"

        if category not in map_data or skill not in map_data[category]:
            return None

        questions = map_data[category][skill].get(difficulty, [])
        if not questions:
            return None

        # Extract asked IDs as a set (O(1) lookup) — Fix #14
        asked_ids = {q["id"] for q in state.get("questions_asked", [])}

        # Filter remaining
        available = [q_id for q_id in questions if q_id not in asked_ids]

        if not available:
            return None

        return random.choice(available)


    DB_PATH = os.getenv('DB_PATH')


    @staticmethod
    def get_question_from_db(question_id: str) -> Optional[QuestionRecord]:
        """
        Fetch the question row for a given question_id and return as QuestionRecord.
        Returns None if no question with that ID exists.
        """
        query = """
        SELECT *
        FROM questions
        WHERE id = ?
        """

        conn = None
        try:
            conn = sqlite3.connect(InterviewAgent.DB_PATH)
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute(query, (question_id,))  # Fix #6: trailing comma for tuple
            row = cursor.fetchone()

            if row is None:
                return None

            # Fix #4: valid return syntax
            new_record: QuestionRecord = {
                "id": row["id"],
                "question": row["question"],
                "user_answer": None
            }
            return new_record

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

        finally:
            if conn:
                conn.close()
                

    @staticmethod
    def generate_question_by_llm(state: InterviewState) -> QuestionRecord:
        """Generate a question using the LLM when no DB question is available."""
        skill = state["skill"]
        category = state["category"]

        prompt = ChatPromptTemplate.from_template(
            "You are technical interviewer of '{category}' in '{skill}'. "
            "Create an easy level interview question in one line"
        )
        chain = prompt | llm
        response = chain.invoke({"category": category, "skill": skill})

        question_id = f"llm-{str(uuid.uuid4())}"

        # Fix #3: valid variable assignment + return
        new_record: QuestionRecord = {
            "id": question_id,
            "question": response.content,
            "user_answer": None
        }
        return new_record

    @staticmethod
    def question_generator_node(state: InterviewState):
        """
        Fetch question from DB (via map.json lookup), or fallback to LLM.
        Completion check is handled in router_logic, not here.
        """
        # Fix #11: correct function name
        q_id = InterviewAgent.retrieve_question_id(state)

        question_content: Optional[QuestionRecord] = None
        if q_id is None:
            question_content = InterviewAgent.generate_question_by_llm(state)
        else:  # Fix #5: added colon
            question_content = InterviewAgent.get_question_from_db(q_id)

        # Fallback if DB lookup also fails
        if question_content is None:
            question_content = InterviewAgent.generate_question_by_llm(state)

        updated_stack = state.get("questions_asked", []) + [question_content]

        return {
            # Fix #9: bracket access on TypedDict, wrap in AIMessage
            "messages": [AIMessage(content=question_content["question"])],
            "questions_asked": updated_stack,
            "is_off_topic": False  # Reset off-topic flag
        }

    @staticmethod
    def record_answer_node(state: InterviewState):
        user_answer = input(f"{state['messages'][-1].content}")

        return {
            "messages": [HumanMessage(content=user_answer)]
        }


    @staticmethod
    def answer_analyzer_node(state: InterviewState):
        """
        Requirement 4: Record answer if valid. Redirect if off-topic.
        - Off-topic: LLM redirects user back to the question. Answer is NOT recorded.
        - Valid answer: Answer is saved into questions_asked for report/scoring.
        """
        last_user_message = state["messages"][-1].content
        stack = state.get("questions_asked", [])

        if not stack:
            return {}  # Safety check

        # 1. Peek at the top of the stack
        last_question_record = stack[-1]

        # 2. Structured validation — guaranteed bool, no string parsing
        structured_llm = llm.with_structured_output(ValidationResult)

        validation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an interview moderator. Your job is to determine if the "
         "candidate's response is an attempt to answer the question OR if "
         "they are going off-topic/chatting casually."),
        ("user",
         "Current Question: {question}\nCandidate Response: {response}")
        ])

        validator_chain = validation_prompt | structured_llm
        result: ValidationResult = validator_chain.invoke({
            "question": last_question_record["question"],
            "response": last_user_message
        })

        if result.is_off_topic:
            # Off-topic: redirect user, do NOT record this as an answer
            redirect_prompt = ChatPromptTemplate.from_template(
                "The candidate said: '{response}'. This is off-topic."
                "Politely but firmly bring them back to the interview context in one line"
                "and re-ask the following question: '{question}'"
            )
            redirect_msg = (redirect_prompt | llm).invoke({
                "response": last_user_message,
                "question": last_question_record["question"]
            })

            return {
                "messages": [redirect_msg],
                "is_off_topic": True
            }
        else:
            # Valid answer: save it into the question record for scoring/report
            last_question_record["user_answer"] = last_user_message
            updated_stack = stack[:-1] + [last_question_record]

            return {
                "questions_asked": updated_stack,
                "is_off_topic": False
            }
        
    @staticmethod
    def completion_node(state: InterviewState):
        """
        Requirement 7: End the interview.
        """

        pprint.pprint(state, width=80, indent=3, sort_dicts=False)
        return {
            "messages": [AIMessage(
                content="Thank you. That concludes our interview session. "
                        "We have recorded your responses."
            )],
            "is_complete": True
        }
    

    @staticmethod
    def router_logic(state: InterviewState):
        """Route after answer analysis: complete, off-topic, or next question.
        
        Completion is based on ANSWERED questions (user_answer is not None),
        not just questions asked. This ensures off-topic loops don't
        prematurely end the interview.
        """
        if state.get("is_off_topic"):
            # Off-topic: loop back to record_answer for the SAME question
            return "off_topic"

        # Count only questions that have been actually answered
        answered = [
            q for q in state.get("questions_asked", [])
            if q.get("user_answer") is not None
        ]
        if len(answered) >= state.get("max_questions", 2):
            return "interview_completed"

        # Valid answer recorded, but more questions to go
        return "ask_again"

    def createworkflow(self):
        workflow = StateGraph(state_schema=InterviewState)

        # 1. Add nodes
        workflow.add_node("pre_interview_node", InterviewAgent.pre_interview_node)
        workflow.add_node("greet", InterviewAgent.greeting_node)
        workflow.add_node("ask_question", InterviewAgent.question_generator_node)
        workflow.add_node("record_answer", InterviewAgent.record_answer_node)
        workflow.add_node("analyze_answer", InterviewAgent.answer_analyzer_node)
        workflow.add_node("finalize", InterviewAgent.completion_node)

        # 2. Edges
        workflow.add_edge(START, "pre_interview_node")
        workflow.add_edge("pre_interview_node", "greet")
        workflow.add_edge("greet", "ask_question")
        workflow.add_edge("ask_question", "record_answer")
        workflow.add_edge("record_answer", "analyze_answer")
        workflow.add_edge("finalize", END)

        # 3. Conditional edges from analyze_answer
        workflow.add_conditional_edges("analyze_answer", InterviewAgent.router_logic, {
            "interview_completed": "finalize",
            "ask_again": "ask_question",
            "off_topic": "record_answer"
        })
        

        # 4. Compile
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)

        return app

config = {"configurable": {"thread_id": "interview_session_001"}}

# Fix #18: removed question_count and interview_record, added questions_asked
initial_state = {
    "max_questions": 2,
    "questions_asked": [],
    "messages": [],
    "is_off_topic": False,
    "is_complete": False
}

print("--- Starting Interview ---")
agent = InterviewAgent()
app = agent.createworkflow()
app.invoke(initial_state, config=config)        