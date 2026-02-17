import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

from question_manager import QuestionManager
from report_generator import ReportGenerator


def build_interview_instructions(questions: List[Dict], category: str, skill: str) -> str:
    """
    Build dynamic system prompt with questions embedded directly.
    This removes the need for a record_answer tool entirely.
    The LLM knows exactly what to ask and when to move on.
    """
    questions_text = "\n".join(
        [f"  Q{i+1}: {q['question']}" for i, q in enumerate(questions)]
    )

    return f"""You are a professional technical interviewer conducting a {skill} interview ({category} category).

You have exactly {len(questions)} questions to ask. Here they are in order:
{questions_text}

== STRICT INTERVIEW PROTOCOL ==

STEP 1 — Ask Q1 immediately. Say: "Let's begin. Question 1: [Q1 text]"
STEP 2 — Stay SILENT and wait. Do NOT speak again until the candidate finishes their answer.
STEP 3 — Once they finish, say ONLY "Thank you." then immediately ask the next question.
         Say: "Question [N]: [QN text]"
STEP 4 — Repeat until all {len(questions)} questions are done.
STEP 5 — After Q{len(questions)} is answered, say: "That completes our interview. Generating your report now."
         Then call end_interview_and_generate_report.

== RULES ==
- Ask questions WORD FOR WORD as written above. Do NOT paraphrase or rewrite them.
- Do NOT give feedback, hints, or reactions beyond "Thank you."
- Do NOT ask follow-up questions.
- Do NOT repeat a question if they ask you to — just re-read it exactly.
- If they go off-topic, gently redirect: "Let's stay focused. [repeat current question]"
- If they ask to stop early, call end_interview_and_generate_report immediately.

Start NOW by asking Question 1."""


class InterviewAssistant(Agent):
    def __init__(self, question_manager: QuestionManager) -> None:
        super().__init__(
            instructions="""You are a professional technical interviewer.

Greet the candidate warmly and ask: "Which technology and skill would you like to be interviewed on today?
For example: Frontend HTML, Backend Python, or Database SQL."

Once they tell you the category and skill, call start_interview immediately.
If they ask what topics are available, call get_available_categories."""
        )
        self.question_manager = question_manager
        self.interview_data: Optional[Dict] = None

    @function_tool
    async def start_interview(
        self, context: RunContext, category: str, skill: str
    ) -> str:
        """Start a technical interview for the given category and skill.

        Args:
            category: The technology category (e.g., 'frontend', 'backend', 'database')
            skill: The specific skill to test (e.g., 'HTML', 'React', 'Python')
        """
        logger.info(f"Starting interview: category={category}, skill={skill}")

        questions = self.question_manager.get_questions(category=category, skill=skill, count=5)

        if not questions:
            return (
                f"No questions found for {category} - {skill}. "
                "Please ask the candidate to choose a different topic."
            )

        self.interview_data = {
            "category": category,
            "skill": skill,
            "start_time": datetime.now().isoformat(),
            "questions": questions,
        }

        # KEY FIX: Embed all questions directly into instructions.
        # The LLM now knows exactly what to ask — no record_answer tool needed.
        self.update_instructions(
            build_interview_instructions(questions, category, skill)
        )

        logger.info(f"Interview started with {len(questions)} questions. Instructions updated.")
        return "Interview ready. Begin asking the questions now."

    @function_tool
    async def end_interview_and_generate_report(
        self, context: RunContext, reason: str = "completed"
    ) -> str:
        """End the interview and generate a comprehensive feedback report.
        Call this after all questions are answered or if the candidate requests to stop.

        Args:
            reason: Why the interview is ending ('completed' or 'user_requested')
        """
        if not self.interview_data:
            return "No active interview found."

        logger.info(f"Ending interview. Reason: {reason}")
        self.interview_data["end_time"] = datetime.now().isoformat()

        # Extract answers from conversation history instead of a broken tool
        answers = self._extract_answers_from_context(context)
        self.interview_data["responses"] = answers

        report = await self._generate_and_save_report()

        self.interview_data = None

        return (
            f"Report generated. Overall score: {report['overall_score']} out of 100. "
            f"{report['summary']} "
            f"Key strengths: {', '.join(report['strengths'][:2])}. "
            f"Areas to work on: {', '.join(report['areas_for_improvement'][:2])}. "
            "The full report has been saved to a file."
        )

    def _extract_answers_from_context(self, context: RunContext) -> List[Dict]:
        """Extract candidate answers from conversation history."""
        if not self.interview_data:
            return []

        questions = self.interview_data["questions"]
        responses = []

        try:
            chat_messages = context.chat_ctx.messages
            user_messages = [m for m in chat_messages if m.role == "user"]
            logger.info(f"Found {len(user_messages)} user messages in conversation")

            for i, q in enumerate(questions):
                # Skip the first user message (topic selection like "Frontend HTML")
                answer_index = i + 1
                if answer_index < len(user_messages):
                    candidate_answer = user_messages[answer_index].content
                    if isinstance(candidate_answer, list):
                        candidate_answer = " ".join(
                            c.text if hasattr(c, "text") else str(c)
                            for c in candidate_answer
                        )
                else:
                    candidate_answer = "[No answer recorded]"

                responses.append({
                    "question_index": i,
                    "question": q["question"],
                    "expected_answer": q["answer"],
                    "candidate_answer": str(candidate_answer),
                })

        except Exception as e:
            logger.error(f"Error extracting answers: {e}")
            for i, q in enumerate(questions):
                responses.append({
                    "question_index": i,
                    "question": q["question"],
                    "expected_answer": q["answer"],
                    "candidate_answer": "[Extraction failed]",
                })

        return responses

    async def _generate_and_save_report(self) -> Dict:
        """Generate report and save to JSON file."""
        report_generator = ReportGenerator()
        report = await report_generator.generate_report(self.interview_data)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category = self.interview_data.get("category", "unknown")
        skill = self.interview_data.get("skill", "unknown")
        filename = f"report_{category}_{skill}_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)

            # Print to console so you can see it immediately
            print(f"\n{'='*60}")
            print(f"INTERVIEW REPORT SAVED: {filename}")
            print(f"{'='*60}")
            print(f"Score:   {report['overall_score']}/100")
            print(f"Summary: {report['summary']}")
            print(f"\nStrengths:")
            for s in report.get("strengths", []):
                print(f"  + {s}")
            print(f"\nAreas to Improve:")
            for a in report.get("areas_for_improvement", []):
                print(f"  - {a}")
            print(f"\nRecommendations:")
            for r in report.get("recommendations", []):
                print(f"  > {r}")
            print(f"{'='*60}\n")
            logger.info(f"Report saved to: {filename}")

        except Exception as e:
            logger.error(f"Failed to save report file: {e}")

        return report

    @function_tool
    async def get_available_categories(self, context: RunContext, filter: str = "all") -> str:
        """Get list of available interview categories and skills.

        Args:
            filter: Optional filter keyword (default: 'all')
        """
        categories = self.question_manager.get_available_categories()
        return f"Available categories and skills: {json.dumps(categories)}"


server = AgentServer()


def prewarm(proc: JobProcess):
    logger.info("Prewarming: Loading VAD and questions...")
    proc.userdata["vad"] = silero.VAD.load()

    excel_path = os.getenv("QUESTIONS_EXCEL_PATH", "questions.xlsx")
    proc.userdata["question_manager"] = QuestionManager(excel_path)
    logger.info("Prewarm complete!")


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    question_manager = ctx.proc.userdata["question_manager"]

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Disabled: was causing LLM to respond before user finished speaking
        preemptive_generation=False,
    )

    await session.start(
        agent=InterviewAssistant(question_manager),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await ctx.connect()

    await session.say(
        "Hello! I'm your AI technical interviewer. "
        "Which technology and skill would you like to be interviewed on today? "
        "For example, you can say Frontend HTML, Backend Python, or Database SQL."
    )


if __name__ == "__main__":
    cli.run_app(server)