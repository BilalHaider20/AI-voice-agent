import pytest
from livekit.agents import AgentSession, inference, llm

from agent import Interviewer


def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


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
                - Mention the name "Alex" as the interviewer's name
                - Reference at least one of the three technical topics:
                  Object-Oriented Programming, Data Structures and Algorithms, or Databases

                The response may include:
                - A warm welcome
                - A brief explanation of the interview format
                - A request for the candidate to introduce themselves
                """,
            )
        )


@pytest.mark.asyncio
async def test_interviewer_asks_oop_after_intro() -> None:
    """After the candidate's self-introduction, the interviewer starts OOP questions."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        # Simulate the candidate's self-introduction
        result = await session.run(
            user_input=(
                "Hi, I'm a software engineer with 3 years of experience. "
                "I've worked mostly in Python and Java, building backend services."
            )
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer's response moves the conversation toward technical questions.

                Acceptable responses include:
                - Acknowledging the introduction and transitioning to technical questions
                - Asking an OOP-related question (e.g., about the four pillars, SOLID, design patterns,
                  inheritance, abstraction, encapsulation, polymorphism, interfaces, abstract classes)
                - Stating that they will begin with Object-Oriented Programming

                The response should NOT:
                - Only ask more personal/background questions without any technical content
                - Be completely off-topic
                """,
            )
        )


@pytest.mark.asyncio
async def test_interviewer_handles_oop_answer() -> None:
    """Interviewer responds constructively when the candidate answers an OOP question."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        # Give the interviewer an OOP answer to respond to
        result = await session.run(
            user_input=(
                "The four pillars of OOP are encapsulation, inheritance, polymorphism, "
                "and abstraction. Encapsulation hides internal state, inheritance lets "
                "classes reuse behavior, polymorphism lets you treat different types "
                "uniformly, and abstraction hides complexity behind simple interfaces."
            )
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer engages constructively with the candidate's OOP answer.

                Acceptable responses include:
                - Acknowledging the answer (e.g., "Good", "Exactly", "That's right")
                - Asking a meaningful follow-up OOP question (e.g., about SOLID principles,
                  design patterns, composition vs inheritance, abstract classes vs interfaces)
                - Probing for a deeper explanation or real-world example

                The response should NOT:
                - Be dismissive or unhelpful
                - Ignore the candidate's answer entirely
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_wraps_up_gracefully() -> None:
    """When the candidate indicates the interview should end, the interviewer wraps up."""
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

                Acceptable responses include:
                - Acknowledging the request and beginning to wrap up
                - Thanking the candidate for their time
                - Mentioning next steps or that the team will follow up

                The response should NOT:
                - Ignore the candidate's request entirely
                - Be rude or dismissive
                """,
            )
        )


@pytest.mark.asyncio
async def test_interviewer_refuses_off_topic_requests() -> None:
    """Interviewer politely stays on topic when asked to do something unrelated."""
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
                The interviewer politely declines to perform tasks outside the interview scope
                and redirects the conversation back to the technical interview.

                The response should NOT:
                - Agree to write a resume or perform unrelated tasks
                - Be rude or unhelpful
                """,
            )
        )

        result.expect.no_more_events()

