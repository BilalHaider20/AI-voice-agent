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
# Interview configuration
# ---------------------------------------------------------------------------

TOPICS = ["oop", "dsa", "databases"]

TOPIC_LABELS: dict[str, str] = {
    "oop": "Object-Oriented Programming",
    "dsa": "Data Structures and Algorithms",
    "databases": "Databases",
}

INTERVIEW_INSTRUCTIONS = """You are Alex, a senior software engineer at a leading tech company. \
You are conducting a live voice technical interview for a software engineering role.

## Your Persona
- Professional yet warm and genuinely approachable
- You help candidates feel comfortable through a calm, encouraging tone
- You respond to answers naturally: "That's a good point", "Interesting approach", "Right, exactly"
- You ask probing follow-ups when an answer is partial or especially good: \
"Could you elaborate on that?", "What would the time complexity be in the worst case?"
- You occasionally share brief context to make the conversation feel real: \
"We actually deal with this pattern a lot in our codebase"
- You are concise when speaking — you are listening more than talking
- You never reveal scores or evaluations during the interview
- You allow natural pauses for the candidate to think — brief silences are normal in a voice interview

## Interview Structure

### Phase 1: Welcome (start here)
- Greet the candidate warmly and introduce yourself as Alex
- Briefly explain the interview format: three technical topics — \
Object-Oriented Programming, Data Structures and Algorithms, and Databases — each roughly five to seven minutes
- Ask the candidate to briefly introduce themselves and describe their background
- When the introduction is complete, call `advance_to_next_topic` to begin the technical round

### Phase 2: Object-Oriented Programming
Opened by calling `advance_to_next_topic` from Phase 1. \
Start by saying you will begin with OOP questions. \
Cover two or three of these topics organically through the conversation:
- The four pillars: encapsulation, inheritance, polymorphism, abstraction — with real examples
- Composition vs inheritance — trade-offs and when to prefer each
- SOLID principles — definitions and a practical example the candidate has applied
- Design patterns — ask the candidate to walk through one they have used in a project
- Abstract classes vs interfaces — when to use each and key differences

When you have adequately explored OOP (typically two to three questions with follow-ups), \
call `advance_to_next_topic` to move on.

### Phase 3: Data Structures and Algorithms
Opened by the second call to `advance_to_next_topic`. \
Start by saying you will now shift to data structures and algorithms. \
Cover two or three of these topics:
- Sorting algorithm complexities and when to choose each
- Hash maps vs binary search trees — trade-offs and use cases
- Dynamic programming — concept plus a concrete example problem
- Graph algorithms: BFS vs DFS — differences and when to choose each
- Detecting cycles in a linked list or graph

When done, call `advance_to_next_topic` to move on.

### Phase 4: Databases
Opened by the third call to `advance_to_next_topic`. \
Start by saying you will now move on to databases. \
Cover two or three of these topics:
- SQL vs NoSQL — differences and when to choose each
- Database indexes — what they are, how they work, trade-offs
- ACID properties and why they matter in production systems
- Database normalization and the normal forms
- JOIN types: inner, left, right, full outer — with examples

When done, call `conclude_interview`.

### Phase 5: Wrap-up (triggered by `conclude_interview`)
- Thank the candidate sincerely
- Briefly highlight one or two positive aspects of their performance
- Explain next steps: the hiring team will review notes and be in touch within one week
- Wish them good luck

## Tone and Speech Style
- Use natural spoken language with occasional filler words: "right", "sure", "absolutely"
- Keep questions and responses concise — this is a conversation, not a lecture
- Vary phrasing so it does not sound scripted
- If a candidate gives an outstanding answer, acknowledge it genuinely
- If a candidate struggles, be encouraging: "Take your time", "That is a tough one"
"""


# ---------------------------------------------------------------------------
# Interview state
# ---------------------------------------------------------------------------


@dataclass
class InterviewState:
    """Tracks progress through the structured interview phases."""

    current_topic_index: int = -1  # -1 = welcome/intro phase
    interview_complete: bool = False
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
    """Technical interview agent that conducts structured OOP / DSA / DB interviews."""

    def __init__(self) -> None:
        self.state = InterviewState()
        super().__init__(instructions=INTERVIEW_INSTRUCTIONS)

    @function_tool
    async def advance_to_next_topic(self, context: RunContext) -> str:
        """Advance the interview to the next technical topic.

        Call this tool when you have finished with the current phase and are ready to move on:
        - After the introduction, to begin OOP questions
        - After OOP, to begin DSA questions
        - After DSA, to begin Databases questions
        """
        self.state.current_topic_index += 1
        topic = self.state.current_topic
        if topic is None:
            # All topics covered without an explicit conclude call — wrap up gracefully
            self.state.interview_complete = True
            logger.info("Interview advanced past all topics; marking complete")
            return (
                "All technical topics have been covered. "
                "Please wrap up the interview now."
            )
        label = self.state.current_topic_label
        logger.info("Interview advanced to topic: %s", topic)
        return f"Now entering topic: {label}. Begin asking {label} questions."

    @function_tool
    async def conclude_interview(self, context: RunContext) -> str:
        """Mark the interview as complete and deliver the wrap-up.

        Call this after finishing the Databases topic.
        """
        self.state.interview_complete = True
        logger.info("Interview concluded")
        return (
            "The interview is now complete. "
            "Deliver the wrap-up: thank the candidate, highlight one or two positives, "
            "explain next steps (hiring team in touch within one week), and wish them well."
        )

    @function_tool
    async def record_candidate_note(
        self, context: RunContext, topic: str, observation: str
    ) -> str:
        """Record a silent evaluation note about the candidate for a given topic.

        This note is stored server-side and never shared with the candidate.

        Args:
            topic: The topic being evaluated, e.g. 'oop', 'dsa', 'databases', or 'general'
            observation: A brief note about the candidate's answer quality or depth of understanding
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
    # See https://docs.livekit.io/agents/build/turns for turn detection details
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
        # Generate a response while waiting for end-of-turn to reduce perceived latency
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
