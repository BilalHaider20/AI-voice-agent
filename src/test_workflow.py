# interactive_test.py
from your_agent_file import InterviewAgent # Change to your filename
from langgraph.checkpoint.memory import MemorySaver
import uuid

def run_interactive_interview():
    # 1. Setup
    agent = InterviewAgent()
    memory = MemorySaver()
    app = agent.createworkflow()
    
    # Create a session ID (thread)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("--- 🎙️ AI INTERVIEW STARTING ---")
    
    # Initial trigger to start the Greeting node
    # We pass an empty dict to kick things off
    events = app.stream({}, config, stream_mode="values")
    
    for event in events:
        # Check if the graph has asked a question or is waiting
        current_stage = event.get("current_stage")
        
        # If we just finished greeting, we need to ask the user for Role/Skill
        if current_stage == "setting_context":
            print("\n[AI]: Hello! I'm your assistant. What Role and Skill are we interviewing for?")
            role = input("Enter Role (e.g. Backend): ")
            skill = input("Enter Skill (e.g. Python): ")
            
            # Inject this into the state and resume
            app.update_state(config, {"category": role, "skill": skill})
            print("\n--- ⏳ Generating your first question... ---")
            
        # If we are in the asking stage, show the question and get the answer
        elif current_stage == "asking_question":
            last_q = event["questions_asked"][-1]
            print(f"\n[AI Question {last_q['question_number']}]: {last_q['question']}")
            
            user_ans = input("[Your Answer]: ")
            
            # Inject the answer so 'save_answer_node' can pick it up
            app.update_state(config, {"user_input": user_ans})

        # Check if the interview is complete
        if event.get("is_complete"):
            print("\n--- 🏁 INTERVIEW COMPLETE ---")
            print("--- 📄 FINAL FEEDBACK REPORT ---")
            # If you added a feedback field in your node, print it here:
            print(event.get("feedback_report", "No report generated yet."))
            break

        # RESUME THE GRAPH
        # After every user input, we call stream(None) to tell the graph to continue
        events = app.stream(None, config, stream_mode="values")

if __name__ == "__main__":
    run_interactive_interview()