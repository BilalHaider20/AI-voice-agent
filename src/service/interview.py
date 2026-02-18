from livekit.agents import Agent
from .interview_state import InterviewState
from .llm_service import get_llm
from langgraph.graph import StateGraph, END
import uuid

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
            current_question_index=0,
            current_stage="greeting",
            is_complete=False,
            max_questions=2,
            questions_asked=[]
        )
        
    def createworkflow(self):
        """Creates the LangGraph workflow for AI interview agent."""
        graph = StateGraph(InterviewState)

        graph.add_node("greeting", self.greeting_node)
        graph.add_node("ask_question", self.ask_question_node)
        graph.add_node("save_answer", self.save_answer_node)  # Add save_answer node
        graph.add_node("end_interview", self.end_interview_node)

        graph.set_entry_point("greeting")
        graph.add_edge("greeting", "ask_question")
        graph.add_edge("ask_question", "save_answer")
        
        # Add conditional edge from save_answer
        graph.add_conditional_edges(
            "save_answer",
            self.should_continue,
            {
                "continue": "ask_question",
                "end": "end_interview"
            }
        )
        
        graph.add_edge("end_interview", END)
        

        return graph.compile(interrupt_before=["save_answer"]) 
        # Added interrupt_before=["save_answer"] so execution pauses 
        # before saving answer, allowing user input to update state externally.

    def should_continue(self, state: InterviewState):
        """Determines if the interview should continue or end."""
        current_index = state.get("current_question_index", 0)
        max_questions = state.get("max_questions", 2) # Default 2 from state
        
        if current_index < max_questions:
            return "continue"
        return "end"

    def greeting_node(self, state: InterviewState):
        """Greets the candidate and asks for skill and category."""

        response = """Hello! I'm your AI Technical Interview Assistant. Please tell me which Skill and Category you would like to be interviewed for?"""

        return {
            "current_stage": "setting_context",
            "questions_asked": [],
        }

    def ask_question_node(self, state: InterviewState):
        """Asks the next question to the candidate."""

        user_skill = state.get("skill")
        user_category = state.get("category")
        
        if not user_skill or not user_category:
             # Fallback if not set, though flow should ensure it.
             user_skill = "General"
             user_category = "Software Engineering"

        llm = get_llm()

        try:
            # Context for the LLM to generate a question
            prompt = f"""
            Generate a technical interview question for a {user_category} role focusing on {user_skill}.
            The question should be concise (one line).
            """
            response = llm.invoke(prompt)
            question_text = response.content.strip()
        except Exception as e:
            question_text = "Tell me about your experience with this technology."

        return {
            "current_stage": "asking_question",
            "questions_asked": [
                {
                    "question_id": str(uuid.uuid4()),
                    "question_number": state["current_question_index"] + 1,
                    "question": question_text,
                    "answer": "", # Expected answer (optional if LLM generates it)
                    "candidate_answer": "",
                    "score": None,
                    "feedback": None,
                }
            ],
            "current_question_index": state["current_question_index"] + 1,
        }

    def save_answer_node(self, state: InterviewState):
        """Saves the candidate's answer to the current question."""
        
        # Get the answer from state (assuming it was injected into state['user_input'] or similar)
        # For this workflow, we assume the caller updates 'candidate_answer' in the state or we process 'user_input'
        
        user_answer = state.get("user_input", "")
        
        questions = state.get("questions_asked", [])
        if questions:
            # Update the last question with the answer
            last_question = questions[-1]
            last_question["candidate_answer"] = user_answer
            # We need to replace the last item or update the list. 
            # In LangGraph/State, returning "questions_asked" usually appends or overwrites depending on reducer.
            # Assuming 'overwrite' or we need to return the full modified list if no reducer.
            # Only InterviewState definition knows reducer. Default TypedDict usually overwrites.
            
            # Let's return the full updated list to be safe if no reducer is defined.
            questions[-1] = last_question
        
        return {
            "questions_asked": questions,
            "current_stage": "answer_received"
        }

    def end_interview_node(self, state: InterviewState):
        """Ends the interview and provides feedback."""

        response = """Thank you for your time. The interview has ended."""

        return {
            "current_stage": "completed",
            "is_complete": True,
        }

    

    