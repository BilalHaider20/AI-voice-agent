"""Generates structured evaluation report from Q&A pairs via LLM."""

import json
import os
from typing import Any

from openai import AsyncOpenAI

from .interview_state import QuestionRecord
from .prompts import format_report_prompt


def _format_qa_pairs(questions: list[QuestionRecord]) -> str:
    lines = []
    for q in questions:
        lines.append(
            f"Q{q['index']}: {q['question']}\n"
            f"  Ground truth: {q['answer']}\n"
            f"  Candidate: {q.get('candidate_answer') or '(no answer)'}"
        )
    return "\n\n".join(lines)


class ReportGenerator:
    """Uses Groq/OpenAI to evaluate answers and produce a strict JSON report."""

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = os.getenv("REPORT_LLM_MODEL", "llama-3.3-70b-versatile")

    async def generate(
        self,
        category: str,
        skill: str,
        questions: list[QuestionRecord],
    ) -> dict[str, Any]:
        """
        Evaluate all answers and return structured report.
        Returns dict with score, summary, strengths, weaknesses, recommendation, question_scores.
        """
        formatted_qa = _format_qa_pairs(questions)
        prompt = format_report_prompt(
            category=category,
            skill=skill,
            formatted_qa_pairs=formatted_qa,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(line for line in lines if not line.startswith("```"))
        report = json.loads(content)

        # Attach per-question scores to state if present
        q_scores = {item["index"]: item for item in report.get("question_scores", [])}
        return {
            "score": report.get("score", 0),
            "summary": report.get("summary", ""),
            "strengths": report.get("strengths", ["", ""])[:2],
            "weaknesses": report.get("weaknesses", ["", ""])[:2],
            "recommendation": report.get("recommendation", ""),
            "question_scores": report.get("question_scores", []),
            "question_scores_by_index": q_scores,
        }
