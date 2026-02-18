"""
LangGraph workflow for interview agent, designed to work with LiveKit LLMAdapter.
Uses MessagesState pattern compatible with LangChain message format.
"""

import logging
import os
from datetime import datetime
import json
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from .interview_state import QuestionRecord
from .question_fetcher import QuestionFetcher
from .report_generator import ReportGenerator

load_dotenv(".env.local")
logger = logging.getLogger("interview_workflow")


class InterviewWorkflowState(MessagesState):
    """State for interview workflow, extends MessagesState for LLMAdapter compatibility."""

    # Interview-specific fields
    category: Optional[str] = None
    skill: Optional[str] = None
    questions: list[QuestionRecord] = []  # noqa: RUF012  # Will be replaced, not merged
    current_index: int = 0
    current_question_text: Optional[str] = None
    last_user_message: Optional[str] = None
    report: Optional[dict] = None
    interview_done: bool = False
    interview_started: bool = False
    next: Optional[str] = None  # For routing decisions


def _get_fetcher() -> QuestionFetcher:
    if not hasattr(_get_fetcher, "_instance"):
        _get_fetcher._instance = QuestionFetcher()
    return _get_fetcher._instance


def call_llm(state: InterviewWorkflowState, llm: ChatOpenAI) -> dict:
    """
    Main LLM node that processes messages and decides next action.
    Uses LLM with tools to make intelligent decisions.
    """
    messages = list(state.get("messages", []))

    # Build system prompt based on interview state
    interview_started = state.get("interview_started", False)
    current_index = state.get("current_index", 0)
    current_question_text = state.get("current_question_text", "")

    if not interview_started:
        # Greeting phase
        system_prompt = (
            "You are a professional technical interviewer conducting a voice interview.\n\n"
            "TASK: Greet the candidate and ask which technology and skill they want to be tested on.\n\n"
            "EXACT GREETING SCRIPT:\n"
            '"Hello. I am your technical interviewer today. Please tell me which technology and skill you would like to be interviewed on. For example: Frontend HTML"\n\n'
            "RULES:\n"
            "- If this is the first message, say the greeting script above.\n"
            "- If the candidate has stated a category and skill, use the begin_interview tool.\n"
            "- Be professional and concise. Wait for their response after greeting."
        )
    elif current_index < len(state.get("questions", [])):
        # Question phase
        question_num = current_index + 1
        system_prompt = (
            f"You are a technical interviewer. You are on question {question_num} of 10.\n\n"
            f"CURRENT QUESTION TO ASK:\n"
            f'"{current_question_text}"\n\n'
            "RULES:\n"
            "- If you haven't asked the current question yet, say: 'Question {question_num}: {current_question_text}'\n"
            "- After asking, wait silently for the candidate's answer.\n"
            "- When the candidate finishes answering, use the capture_answer tool with their complete answer.\n"
            "- If they say 'end interview', 'stop', or 'I want to stop', use the end_interview tool.\n"
            "- Do NOT evaluate answers. Do NOT explain concepts. Just ask questions and record answers."
        )
    else:
        # Report phase
        system_prompt = (
            "The interview is complete. All questions have been answered.\n"
            "Use the generate_report tool to create the final evaluation."
        )

    # Prepare messages with system prompt
    msgs = [SystemMessage(content=system_prompt), *messages]

    # Call LLM
    response = llm.invoke(msgs)

    return {"messages": [response]}


@tool
def begin_interview_tool(category: str, skill: str) -> str:
    """Call this when the candidate has stated which technology and skill they want to be interviewed on.

    Args:
        category: Technology category (e.g. frontend, backend, database).
        skill: Specific skill (e.g. html, python, sql).
    """
    # This will be handled by the tool_executor node
    return f"Starting interview for {category} {skill}"


@tool
def capture_answer_tool(answer: str) -> str:
    """Call this when the candidate has finished answering the current question.

    Args:
        answer: The candidate's complete spoken answer to the current question.
    """
    return f"Answer recorded: {answer[:50]}..."


@tool
def end_interview_tool(reason: str = "") -> str:
    """Call this if the candidate explicitly asks to end or stop the interview early.

    Args:
        reason: Why the interview is ending (optional).
    """
    return "Interview ended early."


def tool_executor_node(state: InterviewWorkflowState) -> dict:
    """Execute tool calls from the LLM's response."""
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    state_updates = {}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        if tool_name == "begin_interview_tool":
            category = tool_args.get("category", "").lower().strip()
            skill = tool_args.get("skill", "").lower().strip()

            # Load questions
            try:
                raw = _get_fetcher().fetch_questions(
                    category=category,
                    skill=skill,
                    count=10,
                    level="easy",
                )

                questions: list[QuestionRecord] = [
                    {
                        "index": i,
                        "question": q["question"],
                        "answer": q["answer"],
                        "answer_keywords": q.get("answer_keywords", ""),
                        "candidate_answer": "",
                        "score": None,
                        "feedback": None,
                    }
                    for i, q in enumerate(raw)
                ]

                current_index = 0
                question_text = questions[current_index]["question"] if questions else ""

                state_updates.update({
                    "category": category,
                    "skill": skill,
                    "questions": questions,
                    "current_index": current_index,
                    "current_question_text": question_text,
                    "interview_started": True,
                })

                result = f"Interview started for {category} {skill}. Question 1: {question_text}"
            except Exception as e:
                logger.exception("Failed to load questions: %s", e)
                result = "Sorry, I could not load questions for that topic. Please try a different category and skill."

        elif tool_name == "capture_answer_tool":
            answer = tool_args.get("answer", "").strip()
            current_index = state.get("current_index", 0)
            questions = list(state.get("questions", []))

            if current_index < len(questions):
                questions[current_index] = {
                    **questions[current_index],
                    "candidate_answer": answer,
                }

                current_index += 1

                if current_index < len(questions):
                    question_text = questions[current_index]["question"]
                    state_updates.update({
                        "questions": questions,
                        "current_index": current_index,
                        "current_question_text": question_text,
                        "last_user_message": answer,
                    })
                    result = f"Answer recorded. Next question: {question_text}"
                else:
                    state_updates.update({
                        "questions": questions,
                        "current_index": current_index,
                        "last_user_message": answer,
                    })
                    result = "All questions answered. Generating report..."
            else:
                result = "No more questions."

        elif tool_name == "end_interview_tool":
            state_updates["interview_done"] = True
            result = "Interview ended early."

        else:
            result = f"Unknown tool: {tool_name}"

        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_name,
                content=result,
            )
        )

    return {
        "messages": tool_messages,
        **state_updates,
    }


def begin_interview_node(state: InterviewWorkflowState) -> dict:
    """Extract category/skill from tool call result and format response."""
    # This node is now handled by tool_executor, but kept for compatibility
    current_index = state.get("current_index", 0)
    question_text = state.get("current_question_text", "")

    if question_text:
        return {
            "messages": [
                AIMessage(content=f"Question {current_index + 1}: {question_text}"),
            ],
        }

    return {"messages": []}


def capture_answer_node(state: InterviewWorkflowState) -> dict:
    """Format response after capturing answer."""
    current_index = state.get("current_index", 0)
    questions = state.get("questions", [])

    if current_index >= len(questions):
        return {"next": "generate_report"}

    question_text = state.get("current_question_text", "")
    if question_text:
        return {
            "messages": [
                AIMessage(content=f"Thank you. Question {current_index + 1}: {question_text}"),
            ],
        }

    return {"messages": []}


async def generate_report_node(state: InterviewWorkflowState) -> dict:
    """Generate final evaluation report."""
    questions = state.get("questions", [])
    category = state.get("category", "")
    skill = state.get("skill", "")

    try:
        generator = ReportGenerator()
        report = await generator.generate(
            category=category,
            skill=skill,
            questions=questions,
        )

        # Save report to file
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/report_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Report saved to {filename}")

        voice_summary = "Interview complete. Your report has been generated. Thank you."

        return {
            "report": report,
            "interview_done": True,
            "messages": [AIMessage(content=voice_summary)],
            "next": "end",
        }
    except Exception as e:
        logger.exception("Report generation failed: %s", e)
        return {
            "messages": [
                AIMessage(content="Interview complete. Thank you for your time.")
            ],
            "next": "end",
        }


def end_interview_node(state: InterviewWorkflowState) -> dict:
    """Handle early interview termination."""
    questions = state.get("questions", [])
    current_index = state.get("current_index", 0)

    if questions and current_index > 0:
        # Generate partial report
        # (Could call generate_report_node here, but simplified for now)
        return {
            "messages": [
                AIMessage(
                    content="Interview ended early. Thank you for your time. We'll generate a report based on the questions answered."
                )
            ],
            "next": "end",
        }
    return {
        "messages": [AIMessage(content="Interview ended. Thank you.")],
        "next": "end",
    }


def decide_next_action(state: InterviewWorkflowState) -> Literal["tool_executor", "begin_interview", "capture_answer", "generate_report", "end_interview", "end"]:
    """Router function for conditional edges."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    # Check if LLM made tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"

    # Check state-based routing
    next_action = state.get("next")
    if next_action == "generate_report":
        return "generate_report"
    if next_action == "end_interview":
        return "end_interview"

    # Check if we need to ask a question
    interview_started = state.get("interview_started", False)
    current_index = state.get("current_index", 0)
    questions = state.get("questions", [])

    if interview_started and current_index < len(questions):
        current_question_text = state.get("current_question_text", "")
        # Check if we just loaded questions (begin_interview was called)
        if current_index == 0 and current_question_text:
            return "begin_interview"
        # Check if we just captured an answer
        if current_index > 0:
            return "capture_answer"

    return "end"


def create_interview_workflow():
    """
    Create LangGraph workflow for interview agent.
    Designed to work with LiveKit's LLMAdapter.
    Initializes LLM with tools for intelligent decision-making.
    """
    # Initialize LLM (Groq compatible)
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    # Define tools
    tools = [begin_interview_tool, capture_answer_tool, end_interview_tool]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create wrapper function for LLM call
    def call_llm_wrapper(state: InterviewWorkflowState) -> dict:
        return call_llm(state, llm_with_tools)

    # Build graph
    graph = StateGraph(InterviewWorkflowState)

    # Add nodes
    graph.add_node("llm", call_llm_wrapper)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("begin_interview", begin_interview_node)
    graph.add_node("capture_answer", capture_answer_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("end_interview", end_interview_node)

    # Set entry point
    graph.set_entry_point("llm")

    # Conditional edges from LLM
    graph.add_conditional_edges(
        "llm",
        decide_next_action,
        {
            "tool_executor": "tool_executor",
            "begin_interview": "begin_interview",
            "capture_answer": "capture_answer",
            "generate_report": "generate_report",
            "end_interview": "end_interview",
            "end": END,
        },
    )

    # From tool_executor back to LLM (to continue conversation)
    graph.add_edge("tool_executor", "llm")

    # Other nodes return to END (workflow completes one turn)
    graph.add_edge("begin_interview", END)
    graph.add_edge("capture_answer", END)
    graph.add_edge("generate_report", END)
    graph.add_edge("end_interview", END)

    # Compile with checkpointer

    return graph.compile(checkpointer=MemorySaver())
