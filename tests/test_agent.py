import pytest
from livekit.agents import AgentSession, inference, llm

from agent import TOPIC_LABELS, Interviewer, InterviewState


def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


# ---------------------------------------------------------------------------
# InterviewState unit tests (pure Python, no LLM needed)
# ---------------------------------------------------------------------------


def test_interview_state_defaults() -> None:
    """InterviewState starts with no topic selected and the interview incomplete."""
    state = InterviewState()
    assert state.selected_topic is None
    assert state.interview_complete is False
    assert state.off_topic_count == 0
    assert state.notes == {}


def test_interview_state_topic_assignment() -> None:
    """selected_topic stores the topic key after assignment."""
    state = InterviewState()
    for key in TOPIC_LABELS:
        state.selected_topic = key
        assert state.selected_topic == key


def test_interview_state_notes_per_topic() -> None:
    """Notes are grouped by topic key and accumulate correctly."""
    state = InterviewState()
    state.notes.setdefault("oop", []).append("strong encapsulation answer")
    state.notes.setdefault("oop", []).append("missed polymorphism example")
    state.notes.setdefault("dsa", []).append("good BFS explanation")
    assert state.notes["oop"] == [
        "strong encapsulation answer",
        "missed polymorphism example",
    ]
    assert state.notes["dsa"] == ["good BFS explanation"]


def test_interview_state_off_topic_count() -> None:
    """off_topic_count increments independently of other fields."""
    state = InterviewState()
    state.off_topic_count += 1
    assert state.off_topic_count == 1
    state.off_topic_count += 1
    assert state.off_topic_count == 2


# ---------------------------------------------------------------------------
# Phase 1 — Welcome
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_introduces_itself() -> None:
    """Interviewer greets the candidate as Alex and mentions the topic-selection format."""
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
                - Indicate that the candidate will choose or select a topic,
                  OR reference at least one of the three available topics:
                  Object-Oriented Programming, Data Structures and Algorithms, or Databases

                The response may include:
                - A warm welcome
                - A brief description of the session format (one topic of the candidate's choice)
                - A request for the candidate to introduce themselves
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Phase 2 — Topic selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interviewer_asks_candidate_to_select_topic() -> None:
    """After the candidate's introduction, the interviewer asks them to choose a topic."""
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
                The interviewer asks the candidate to choose one interview topic.

                The response must:
                - Ask the candidate which topic they would like to focus on
                - Name at least one of the three topics: Object-Oriented Programming,
                  Data Structures and Algorithms, or Databases

                The response must NOT:
                - Jump straight into technical questions without asking the candidate
                  to select a topic first
                - Be completely unrelated to software engineering
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_starts_oop_after_selection() -> None:
    """When the candidate says they want OOP, the interviewer asks an OOP question."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="I'd like to do Object-Oriented Programming please."
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer acknowledges the topic choice and begins OOP questions.

                Acceptable responses:
                - Confirming the topic is OOP and asking an OOP question
                - Asking about the four pillars of OOP
                - Asking about SOLID principles, design patterns, composition vs
                  inheritance, or abstract classes vs interfaces
                - Presenting a scenario-based OOP question

                The response must NOT:
                - Ask a DSA or Databases question
                - Ignore the candidate's topic selection and ask about a different topic
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_starts_dsa_after_selection() -> None:
    """When the candidate says they want DSA, the interviewer asks a DSA question."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="Let's go with Data Structures and Algorithms."
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer acknowledges the topic choice and begins DSA questions.

                Acceptable responses:
                - Confirming the topic is DSA / data structures and algorithms
                - Asking about sorting algorithms, hash maps vs trees, dynamic
                  programming, BFS vs DFS, or a scenario-based DSA question

                The response must NOT:
                - Ask an OOP or Databases question
                - Ignore the candidate's topic selection
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_starts_databases_after_selection() -> None:
    """When the candidate says they want Databases, the interviewer asks a DB question."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(user_input="I want to do Databases.")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer acknowledges the topic choice and begins Databases questions.

                Acceptable responses:
                - Confirming the topic is Databases
                - Asking about SQL vs NoSQL, database indexes, ACID properties,
                  normalization, or a scenario-based database question

                The response must NOT:
                - Ask an OOP or DSA question
                - Ignore the candidate's topic selection
                """,
            )
        )

        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_interviewer_clarifies_ambiguous_topic_selection() -> None:
    """When the candidate gives an unclear topic choice, the interviewer asks to clarify."""
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Interviewer())

        result = await session.run(
            user_input="Hmm, maybe something with systems? I'm not sure."
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="""
                The interviewer does not start technical questions because the topic
                selection was unclear. Instead it asks the candidate to choose again.

                The response must:
                - Ask the candidate to choose one of the three specific topics:
                  Object-Oriented Programming, Data Structures and Algorithms,
                  or Databases

                The response must NOT:
                - Start asking any technical interview questions
                - Choose a topic on behalf of the candidate without their confirmation
                """,
            )
        )

        result.expect.no_more_events()


# ---------------------------------------------------------------------------
# Phase 3 — OOP follow-up probing (topic-agnostic behavior)
# ---------------------------------------------------------------------------


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
