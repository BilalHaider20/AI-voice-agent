import logging
from service.interview import InterviewAgent
from dotenv import load_dotenv
from livekit import rtc
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero, deepgram, groq, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
import os

load_dotenv(".env.local")


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the voice agent."""
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=qroq.LLM(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    interview_agent = InterviewAgent()

    await session.start(
        room=ctx.room,
        agent=interview_agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False  # optional: keep session even if participant disconnects
        ),
    )

    await ctx.connect()

    # Initial greeting
    await session.generate_reply(
        instructions="Hello! I'm your AI interview assistant."
    )

    # Handle user messages (sync callback scheduling async work)
    # def on_user_speech(message):
    #     asyncio.create_task(travel_agent.process_user_input(message.text, session))
    # session.on("user_speech_committed", on_user_speech)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))