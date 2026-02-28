import pytest
from livekit.agents import AgentSession, inference, llm

from agent import Interviewer


def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Phase 1 — Welcome
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_introduces_itself() -> None:
    """Interviewer greets the candidate, introduces itself as Alex, and explains the format."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(user_input="Hello, I'm ready for the interview.")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                Greets the candidate and introduces itself as "Alex".

                The response must:
                - Mention the interviewer's name "Alex"
                - Reference at least one of the three technical topics:
                  Object-Oriented Programming, Data Structures and Algorithms, or Databases

                The response may include:
                - A warm welcome
                - A brief description of the interview format or structure
                - A request for the candidate to introduce themselves
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Phase 2 — OOP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_asks_oop_after_intro() -> None:
    """After the candidate's self-introduction, the interviewer transitions to OOP questions."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input=(
                "Hi, I'm a software engineer with 3 years of experience. "
                "I've worked mostly in Python and Java building backend services."
            )
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer transitions toward technical interview questions.

                Acceptable responses:
                - Acknowledging the introduction and announcing the start of technical questions
                - Asking an OOP-related question such as the four pillars, SOLID principles,
                  design patterns, inheritance, abstraction, encapsulation, polymorphism,
                  interfaces, abstract classes, or composition vs inheritance
                - Presenting a scenario-based OOP question (system design involving OOP concepts)

                The response must NOT:
                - Only ask more personal background questions without any technical content
                - Be completely unrelated to OOP or software engineering
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_probes_follow_up_on_oop_answer() -> None:
    """Interviewer asks a follow-up to deepen a correct but surface-level OOP answer."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input=(
                "The four pillars of OOP are encapsulation, inheritance, "
                "polymorphism, and abstraction."
            )
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer engages constructively and probes deeper.

                Acceptable responses:
                - Acknowledging the answer and asking for a real-world example
                - Asking about a specific pillar in more depth
                - Asking a follow-up OOP question (SOLID, design patterns, composition vs
                  inheritance, abstract classes vs interfaces)
                - Presenting a scenario question based on OOP concepts

                The response must NOT:
                - Accept the answer and immediately move to a completely different topic
                  without any follow-up engagement
                - Be dismissive or unhelpful
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Clarification behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_asks_clarification_on_vague_answer() -> None:
    """Interviewer requests clarification when the candidate gives an overly vague answer."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="It just makes things faster and more scalable, you know?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer asks for clarification or more specificity.

                Acceptable responses:
                - Asking what "faster" means in this context (time complexity? throughput?)
                - Asking what "more scalable" means specifically
                - Asking the candidate to elaborate or be more precise
                - Asking the candidate to give a concrete example

                The response must NOT:
                - Accept the vague answer and move on without any clarification attempt
                - Be rude or dismissive
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Guidance / hint behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_guides_stuck_candidate() -> None:
    """Interviewer offers a graduated hint when the candidate admits they do not know."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input=(
                "Hmm, I'm not really sure about this one. "
                "I can't remember how hash maps work internally."
            )
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer is encouraging and offers help to the stuck candidate.

                Acceptable responses:
                - Offering encouragement such as "Take your time" or "That's okay"
                - Providing a hint or analogy (e.g. comparing a hash map to a real-life
                  dictionary or filing cabinet)
                - Reframing the question to make it easier to approach
                - Asking a simpler sub-question to guide the candidate

                The response must NOT:
                - Simply give the complete answer directly without any prompting or hint
                - Be discouraging or dismissive
                - Move immediately to the next question without any attempt to help
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Off-topic detection and redirection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_redirects_off_topic_resume_request() -> None:
    """Interviewer declines out-of-scope requests and redirects to the interview."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="Can you help me write my resume instead of interviewing me?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer politely declines the off-topic request and redirects
                the conversation back to the technical interview.

                The response must NOT:
                - Agree to write a resume or perform any task unrelated to the interview
                - Be rude or dismissive

                The response should:
                - Acknowledge the request briefly without being harsh
                - Redirect back to the interview topics
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_does_not_reveal_score() -> None:
    """Interviewer refuses to share evaluation scores mid-interview."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="How am I doing so far? What score would you give me?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer declines to share a score or evaluation mid-interview.

                Acceptable responses:
                - Saying feedback or scores are shared after the interview is complete
                - Reassuring the candidate without revealing any score
                - Redirecting back to the interview questions

                The response must NOT:
                - Provide any numerical score or explicit performance rating
                - State that the candidate is doing poorly or very well in evaluative terms
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_wraps_up_gracefully() -> None:
    """Interviewer wraps up with thanks, highlights, and next steps when asked to conclude."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="I think we've covered everything. Can we wrap up now?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer responds to the candidate's request to end the session.

                Acceptable responses:
                - Acknowledging the request and beginning to wrap up
                - Thanking the candidate for their time
                - Mentioning next steps or that the team will follow up

                The response must NOT:
                - Ignore the candidate's request
                - Be rude or dismissive
                """,
            )
        )

        result.expect.no_more_events()
