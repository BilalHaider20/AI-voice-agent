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
# Topic registry
# ---------------------------------------------------------------------------

# Valid topic keys used as arguments to select_topic().
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
• In one sentence explain the format: this session will focus on one technical topic \
of their choice — Object-Oriented Programming, Data Structures and Algorithms, or Databases.
• Ask the candidate to briefly introduce themselves and share their background.

PHASE 2 — TOPIC SELECTION (after the candidate finishes their introduction)
• Ask: "Which topic would you like to focus on today — Object-Oriented Programming, \
Data Structures and Algorithms, or Databases?"
• Listen carefully to their response and map it to the correct key:
  - "OOP", "object-oriented", "object oriented programming", "classes" → key: 'oop'
  - "DSA", "data structures", "algorithms", "data structures and algorithms" → key: 'dsa'
  - "databases", "database", "SQL", "NoSQL", "db" → key: 'databases'
• Once they clearly choose one topic, call `select_topic` with the matching key.
• If their choice is unclear or does not match one of the three options, do NOT call \
`select_topic` — instead ask them to choose again:
  "I want to make sure I pick the right topic — could you say which one: \
Object-Oriented Programming, Data Structures and Algorithms, or Databases?"
• Do NOT ask any technical questions until `select_topic` has been successfully called.

PHASE 3 — TOPIC INTERVIEW (entered once `select_topic` returns successfully)
Open by saying: "Great, let's get into [topic name]." \
Then ask two or three questions from the matching section below. \
Mix at least one concept question with at least one scenario question. \
Apply all conversational behaviours (follow-ups, clarification, hints). \
When the topic has been covered adequately, call `conclude_interview`.

────────────────────────────────────────
OOP — Object-Oriented Programming
────────────────────────────────────────
CONCEPT QUESTIONS (choose one or two):
• "Can you walk me through the four pillars of OOP and give me a real-world example \
of each one?"
• "What is the difference between composition and inheritance? When would you favour \
composition over inheritance?"
• "Can you explain the SOLID principles? Pick one that you have actively applied and \
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

────────────────────────────────────────
DSA — Data Structures and Algorithms
────────────────────────────────────────
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

────────────────────────────────────────
DATABASES
────────────────────────────────────────
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

PHASE 4 — WRAP-UP (triggered by `conclude_interview`)
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
    """Tracks the state of a single-topic voice interview session.

    ``selected_topic`` is ``None`` until the candidate chooses a topic via
    ``select_topic``.  Once set it is one of the keys in ``TOPIC_LABELS``
    (``'oop'``, ``'dsa'``, or ``'databases'``).

    Known Limitations
    -----------------
    * Only one topic per session — multi-topic ordering is not supported.
    * No difficulty level (junior / senior) is recorded or used.
    * ``notes`` are in-memory only; they are lost when the session ends.
    * No per-question deduplication — the same question could theoretically
      be asked twice within a session.
    * No time-limit enforcement per topic.
    """

    selected_topic: Optional[str] = None  # None = topic not yet chosen
    interview_complete: bool = False
    off_topic_count: int = 0
    notes: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Interviewer(Agent):
    """Production-grade technical interview agent.

    The candidate selects one topic (OOP, DSA, or Databases) at the start of
    the session.  The agent then conducts a focused interview on that topic,
    handling off-topic requests, clarification prompts, follow-up probing,
    and graduated hints for stuck candidates.
    """

    def __init__(self) -> None:
        self.state = InterviewState()
        super().__init__(instructions=INTERVIEW_INSTRUCTIONS)

    # ------------------------------------------------------------------
    # Phase management tools
    # ------------------------------------------------------------------

    @function_tool
    async def select_topic(self, context: RunContext, topic: str) -> str:
        """Record the candidate's chosen interview topic and begin that topic's questions.

        Call this once the candidate has clearly stated which topic they want.
        If their choice is ambiguous, do NOT call this tool — ask them to
        choose again, then call it once they clarify.

        Handles topic changes: if the candidate changes their mind before any
        questions have been asked, this tool accepts the new choice and returns
        updated instructions.

        Args:
            topic: Must be exactly one of: 'oop', 'dsa', 'databases'
        """
        normalized = topic.lower().strip()
        if normalized not in TOPIC_LABELS:
            logger.warning("select_topic called with invalid key %r", topic)
            return (
                "Invalid topic key — do not use this return value to start questions. "
                "Ask the candidate to choose one of: Object-Oriented Programming, "
                "Data Structures and Algorithms, or Databases. "
                "Then call select_topic again with 'oop', 'dsa', or 'databases'."
            )

        label = TOPIC_LABELS[normalized]

        if (
            self.state.selected_topic is not None
            and self.state.selected_topic != normalized
            and not self.state.interview_complete
        ):
            old_label = TOPIC_LABELS[self.state.selected_topic]
            self.state.selected_topic = normalized
            logger.info("Candidate changed topic: %s → %s", old_label, label)
            return (
                f"Topic changed from {old_label} to {label}. "
                f"Acknowledge the change briefly and begin the {label} interview now."
            )

        self.state.selected_topic = normalized
        logger.info("Candidate selected topic: %s", normalized)
        return (
            f"Topic selected: {label}. "
            f"Open by saying 'Great, let's get into {label}.' "
            f"Then ask two or three questions from the {label} section."
        )

    @function_tool
    async def conclude_interview(self, context: RunContext) -> str:
        """Mark the interview as complete and deliver the wrap-up.

        Call this after finishing all questions for the selected topic.
        """
        self.state.interview_complete = True
        logger.info(
            "Interview concluded (topic: %s)", self.state.selected_topic or "unknown"
        )
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
            topic: The current interview topic or phase, e.g. 'oop', 'selection', 'general'
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
