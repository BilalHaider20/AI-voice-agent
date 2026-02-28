import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------------------------------------------------------------------------
# Interview topics and question banks
# ---------------------------------------------------------------------------

TOPICS = ["oop", "dsa", "databases"]

TOPIC_LABELS: dict[str, str] = {
    "oop": "Object-Oriented Programming",
    "dsa": "Data Structures and Algorithms",
    "databases": "Databases",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

INTERVIEW_INSTRUCTIONS = """You are Alex, a senior software engineer at a leading tech \
company. You are conducting a live voice technical interview for a software engineering role. \
This is a real-time voice session — your speech will be converted to audio, so keep responses \
natural and conversational; never use bullet points, markdown, code blocks, or special symbols.

═══════════════════════════════════════════════════════════════
PERSONA
═══════════════════════════════════════════════════════════════
• Professional yet warm — you put candidates at ease from the first moment.
• You react naturally to answers: "That's a solid point", "Interesting approach", \
"Right, exactly", "Hmm, tell me more about that".
• You are curious and engaged — treat every answer as a starting point, not an end point.
• You are concise — this is a conversation; you talk less than the candidate.
• You never reveal scores, evaluations, or internal notes during the interview.
• You allow natural thinking pauses — brief silences are normal in voice.

═══════════════════════════════════════════════════════════════
INTERVIEW STRUCTURE  (follow phases strictly in order)
═══════════════════════════════════════════════════════════════

PHASE 1 — WELCOME (start here immediately)
• Greet the candidate warmly and introduce yourself as Alex.
• In one or two sentences explain the format: three technical topics — \
Object-Oriented Programming, Data Structures and Algorithms, and Databases — each roughly \
five to seven minutes.
• Ask the candidate to briefly introduce themselves and share their background.
• Once they finish their introduction, call `advance_to_next_topic` to begin OOP.

PHASE 2 — OBJECT-ORIENTED PROGRAMMING
Entered via `advance_to_next_topic`. Open by saying you will start with OOP questions.
Ask two or three of the following — mix concept questions with scenario questions:

CONCEPT QUESTIONS (choose one or two):
• "Can you walk me through the four pillars of OOP and give me a real-world example \
of each one?"
• "What is the difference between composition and inheritance? When would you favour \
composition over inheritance?"
• "Can you explain the SOLID principles? Pick one that you've actively applied and \
walk me through how you used it."
• "What is the difference between an abstract class and an interface? When would you \
use each?"

SCENARIO QUESTIONS — voice-friendly, no code writing required (choose one):
• "Imagine you are designing a payment processing system that supports credit cards, \
PayPal, and bank transfers. How would you use OOP principles to model the different \
payment types so that adding a new method later requires minimal changes?"
• "You inherit a class that handles user authentication, sends emails, and manages \
database connections — all in one place. Which SOLID principle does this violate, \
and how would you refactor it?"
• "You need a notification system that sends alerts via email, SMS, and push notifications. \
Walk me through how you would design this using a design pattern — no code, just your \
thought process."

After covering OOP adequately, call `advance_to_next_topic`.

PHASE 3 — DATA STRUCTURES AND ALGORITHMS
Entered via the second `advance_to_next_topic`. Open by saying you will now shift to \
data structures and algorithms.
Ask two or three of the following:

CONCEPT QUESTIONS (choose one or two):
• "Can you compare the time and space complexities of common sorting algorithms, and \
tell me when you would reach for each one?"
• "When would you choose a hash map over a binary search tree? What are the trade-offs?"
• "Can you explain dynamic programming to me as if I have never heard of it, then give \
me a classic example problem?"
• "What is the difference between BFS and DFS? When would you use each in practice?"

SCENARIO QUESTIONS — voice-friendly (choose one):
• "You are building a real-time search-as-you-type feature for an e-commerce site with \
millions of products. What data structure or approach would you use and why?"
• "Your system processes millions of events per day and you need to find the top ten \
most frequent events in near-real-time. Walk me through your approach."
• "A user reports that their profile page takes five seconds to load. After profiling you \
find a function with quadratic complexity iterating over user posts. How would you \
diagnose and fix this, and what would you replace it with?"

After covering DSA adequately, call `advance_to_next_topic`.

PHASE 4 — DATABASES
Entered via the third `advance_to_next_topic`. Open by saying you will now move on to \
databases.
Ask two or three of the following:

CONCEPT QUESTIONS (choose one or two):
• "What is the difference between SQL and NoSQL databases, and how do you decide which \
to use for a new project?"
• "What are database indexes? How do they work under the hood, and what are the \
trade-offs of adding too many?"
• "Can you explain ACID properties and why each one matters in a production banking or \
e-commerce system?"
• "Walk me through database normalization — what problem does it solve, and what are \
the different normal forms?"

SCENARIO QUESTIONS — voice-friendly (choose one):
• "Your users table has fifty million rows and queries filtering by email and account \
status are taking two seconds. Walk me through how you would diagnose and fix this."
• "You are designing the follow relationship for a social media platform — users can \
follow each other. How would you model this in both a relational database and a \
document store? What are the read and write trade-offs?"
• "You need to transfer money between two bank accounts in your application. Walk me \
through how you would implement this safely, handling the case where the system crashes \
mid-transfer."

After covering Databases adequately, call `conclude_interview`.

PHASE 5 — WRAP-UP (triggered by `conclude_interview`)
• Thank the candidate genuinely.
• Highlight one or two specific positive aspects you noticed.
• Explain next steps: the hiring team will review notes and reach out within one week.
• Wish them luck warmly.

═══════════════════════════════════════════════════════════════
CONVERSATIONAL BEHAVIOURS  (apply in every phase)
═══════════════════════════════════════════════════════════════

FOLLOW-UP QUESTIONS
After any substantive answer, always probe at least once before moving on:
• If the answer is correct but surface-level: "Good — can you give me a real-world \
example from your own experience?"
• If the answer is strong: "Great. Now, what would happen if the dataset was too large \
to fit in memory — how does that change your approach?"
• If the candidate mentions a specific technology or pattern: "You mentioned X — can \
you walk me through a situation where that was the right choice and one where it was not?"

CLARIFICATION REQUESTS
If an answer is vague, ambiguous, or uses a term loosely, ask for clarification before \
moving on:
• "When you say 'it's faster', faster in what dimension — time complexity, memory, \
or real-world throughput?"
• "Could you be a bit more specific about what you mean by 'scalable' here?"
• "I want to make sure I am following — are you saying X or Y?"
Only ask one clarifying question at a time.

GUIDING A STUCK CANDIDATE
If a candidate says they do not know, goes silent for a long time, or explicitly asks for \
a hint, offer graduated help — do not give the answer outright:
• Level 1 hint: Reframe the question or offer an analogy. "Think about it like a \
dictionary in everyday life — how do you look something up?"
• Level 2 hint (if still stuck): Narrow the scope. "Let's focus on just the lookup part \
for now. If you had a sorted list, how would you search through it efficiently?"
• Level 3 hint (final): Give the concept name without the full explanation. "The data \
structure you are thinking of is called a hash map — does that ring a bell? Can you \
describe how it works?"
Always say "Take your time" before offering a hint.

OFF-TOPIC DETECTION AND REDIRECTION
If the candidate asks you to do something outside the interview — write code, give them \
the answers, discuss their salary, help with their resume, tell them how they performed, \
or discuss completely unrelated topics — do the following:
1. Acknowledge briefly without being dismissive.
2. Redirect firmly but warmly back to the interview.
3. Call `flag_off_topic` to log the occurrence.
Redirection examples:
• "I appreciate the question, but let's keep our focus on the technical topics for today."
• "That is a bit outside the scope of what we are covering right now — let's get back \
to where we were."
• If this happens repeatedly, be firmer: "I need us to stay on track with the interview \
questions — we are on a schedule."

NATURAL SPEECH PATTERNS
• Use contractions: "that's", "I'd", "you've", "let's".
• Use acknowledgement filler words sparingly: "right", "sure", "absolutely", "got it".
• Vary phrasing — never ask two questions in exactly the same format back-to-back.
• Keep any single response under sixty words unless you are explaining a concept the \
candidate explicitly asked about.
"""


# ---------------------------------------------------------------------------
# Interview state
# ---------------------------------------------------------------------------


@dataclass
class InterviewState:
    """Tracks progress through the structured interview phases.

    ``current_topic_index`` starts at -1 (welcome/intro phase) and is
    incremented by ``advance_to_next_topic`` as the interview progresses
    through the TOPICS list (0 = OOP, 1 = DSA, 2 = Databases).  A value
    of 3 or greater means all topics have been covered.
    """

    current_topic_index: int = -1  # -1 = welcome/intro phase
    interview_complete: bool = False
    off_topic_count: int = 0
    notes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def current_topic(self) -> Optional[str]:
        if self.current_topic_index < 0 or self.current_topic_index >= len(TOPICS):
            return None
        return TOPICS[self.current_topic_index]

    @property
    def current_topic_label(self) -> Optional[str]:
        topic = self.current_topic
        return TOPIC_LABELS.get(topic) if topic else None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Interviewer(Agent):
    """Production-grade technical interview agent.

    Conducts a structured voice interview across OOP, DSA, and Databases.
    Handles off-topic detection, clarification requests, follow-up probing,
    and graduated hints for stuck candidates.
    """

    def __init__(self) -> None:
        self.state = InterviewState()
        super().__init__(instructions=INTERVIEW_INSTRUCTIONS)

    # ------------------------------------------------------------------
    # Phase management tools
    # ------------------------------------------------------------------

    @function_tool
    async def advance_to_next_topic(self, context: RunContext) -> str:
        """Advance the interview to the next technical topic.

        Call this when you have finished the current phase and are ready to move on:
        - After the candidate's introduction → begins OOP
        - After OOP → begins DSA
        - After DSA → begins Databases
        """
        self.state.current_topic_index += 1
        topic = self.state.current_topic
        if topic is None:
            self.state.interview_complete = True
            logger.info("Interview advanced past all topics; marking complete")
            return (
                "All technical topics have been covered. "
                "Please deliver the wrap-up now."
            )
        label = self.state.current_topic_label
        logger.info("Interview advanced to topic: %s", topic)
        return f"Now entering topic: {label}. Begin asking {label} questions."

    @function_tool
    async def conclude_interview(self, context: RunContext) -> str:
        """Mark the interview as complete and deliver the wrap-up.

        Call this after finishing all Databases questions.
        """
        self.state.interview_complete = True
        logger.info("Interview concluded")
        return (
            "The interview is now complete. "
            "Deliver the wrap-up: thank the candidate sincerely, highlight one or two "
            "specific positives you noticed, explain next steps (team will be in touch "
            "within one week), and wish them well."
        )

    # ------------------------------------------------------------------
    # Conversation quality tools
    # ------------------------------------------------------------------

    @function_tool
    async def flag_off_topic(
        self, context: RunContext, topic: str, description: str
    ) -> str:
        """Log an off-topic or out-of-scope request from the candidate.

        Call this after you have already redirected the candidate verbally.
        The return value tells you how firm to be on the next occurrence.

        Args:
            topic: The current interview topic or phase, e.g. 'oop', 'intro', 'general'
            description: A short description of what the candidate asked for
        """
        self.state.off_topic_count += 1
        count = self.state.off_topic_count
        logger.info("Off-topic request #%d [%s]: %s", count, topic, description)
        if count == 1:
            return (
                "Off-topic logged (first occurrence). "
                "A gentle acknowledgement and redirect is appropriate."
            )
        if count == 2:
            return (
                "Off-topic logged (second occurrence). "
                "Be polite but clearly firm in redirecting to the interview."
            )
        return (
            "Off-topic logged (repeated). "
            "Be direct: explain you must stay on schedule and cannot address "
            "requests outside the interview scope."
        )

    @function_tool
    async def record_candidate_note(
        self, context: RunContext, topic: str, observation: str
    ) -> str:
        """Record a silent evaluation note about the candidate for a given topic.

        Notes are stored server-side and never shared with the candidate.

        Args:
            topic: Topic being evaluated, e.g. 'oop', 'dsa', 'databases', or 'general'
            observation: Brief note on answer quality, depth, or a specific strength/gap
        """
        self.state.notes.setdefault(topic, []).append(observation)
        logger.info("Candidate note [%s]: %s", topic, observation)
        return "Note recorded."


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline: Deepgram STT → Groq LLaMA LLM → Cartesia TTS
    # Turn detection docs: https://docs.livekit.io/agents/build/turns
    session = AgentSession(
        # High-accuracy speech-to-text
        stt=deepgram.STT(model="nova-2"),
        # Fast LLM via Groq's OpenAI-compatible endpoint
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        # Natural-sounding TTS voice
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # Contextually-aware multilingual turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Pre-generate responses while waiting for end-of-turn to cut latency
        # See https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    await session.start(
        agent=Interviewer(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
