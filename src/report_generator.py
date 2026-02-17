import logging
import os
from typing import Dict, List
from datetime import datetime
import json
import openai

logger = logging.getLogger("report_generator")


class ReportGenerator:
    """
    Generates comprehensive interview feedback reports by:
    1. Comparing candidate answers with ground truth
    2. Using LLM to evaluate technical accuracy and completeness
    3. Generating overall performance summary and recommendations
    """

    def __init__(self):
        """Initialize the report generator with LLM client."""
        # Use Groq for fast report generation (same as your agent)
        self.client = openai.OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.3-70b-versatile"

    async def generate_report(self, interview_data: Dict) -> Dict:
        """
        Generate a comprehensive feedback report for the interview.

        Args:
            interview_data: Dictionary containing:
                - category: Interview category
                - skill: Skill being tested
                - start_time: Interview start timestamp
                - end_time: Interview end timestamp
                - questions: List of questions asked
                - responses: List of candidate responses with ground truth

        Returns:
            Dictionary containing the full report with:
                - summary: Brief overall assessment
                - question_evaluations: Detailed feedback for each question
                - overall_score: Numeric score (0-100)
                - strengths: List of strengths
                - areas_for_improvement: List of areas to improve
                - recommendations: Learning recommendations
        """
        logger.info("Generating interview report...")

        # Evaluate each question-answer pair
        evaluations = []
        total_score = 0

        for response in interview_data["responses"]:
            evaluation = await self._evaluate_answer(
                question=response["question"],
                expected_answer=response["expected_answer"],
                candidate_answer=response["candidate_answer"],
            )
            evaluations.append(evaluation)
            total_score += evaluation["score"]

        # Calculate overall score
        num_questions = len(interview_data["responses"])
        overall_score = (total_score / num_questions) if num_questions > 0 else 0

        # Generate overall assessment
        overall_assessment = await self._generate_overall_assessment(
            interview_data, evaluations, overall_score
        )

        # Compile the full report
        report = {
            "interview_metadata": {
                "category": interview_data["category"],
                "skill": interview_data["skill"],
                "start_time": interview_data["start_time"],
                "end_time": interview_data.get("end_time"),
                "num_questions": num_questions,
            },
            "overall_score": round(overall_score, 1),
            "summary": overall_assessment["summary"],
            "strengths": overall_assessment["strengths"],
            "areas_for_improvement": overall_assessment["areas_for_improvement"],
            "recommendations": overall_assessment["recommendations"],
            "question_evaluations": evaluations,
            "generated_at": datetime.now().isoformat(),
        }

        logger.info(f"Report generated. Overall score: {overall_score}/100")
        return report

    async def _evaluate_answer(
        self, question: str, expected_answer: str, candidate_answer: str
    ) -> Dict:
        """
        Evaluate a single answer using LLM.

        Returns:
            Dictionary with score (0-100), feedback, and key points
        """
        prompt = f"""You are an expert technical interviewer evaluating a candidate's answer.

Question: {question}

Expected Answer (Ground Truth): {expected_answer}

Candidate's Answer: {candidate_answer}

Evaluate the candidate's answer based on:
1. Technical accuracy
2. Completeness (did they cover the key points?)
3. Clarity and understanding

Provide your evaluation in JSON format:
{{
    "score": <0-100>,
    "feedback": "<2-3 sentences of specific feedback>",
    "covered_points": ["<key point 1>", "<key point 2>"],
    "missed_points": ["<missed point 1>", "<missed point 2>"],
    "accuracy": "<correct/partially_correct/incorrect>"
}}

Be fair but thorough. Award partial credit for partially correct answers."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical interviewer. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            # Parse JSON response
            evaluation = json.loads(response.choices[0].message.content)

            # Add the question and answers to the evaluation
            evaluation["question"] = question
            evaluation["expected_answer"] = expected_answer
            evaluation["candidate_answer"] = candidate_answer

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            # Fallback evaluation
            return {
                "question": question,
                "expected_answer": expected_answer,
                "candidate_answer": candidate_answer,
                "score": 50,
                "feedback": "Unable to evaluate automatically. Please review manually.",
                "covered_points": [],
                "missed_points": [],
                "accuracy": "unknown",
            }

    async def _generate_overall_assessment(
        self, interview_data: Dict, evaluations: List[Dict], overall_score: float
    ) -> Dict:
        """Generate overall assessment, strengths, and recommendations."""
        prompt = f"""You are an expert technical interviewer providing final feedback for a candidate.

Interview Details:
- Category: {interview_data['category']}
- Skill: {interview_data['skill']}
- Overall Score: {overall_score}/100

Individual Question Evaluations:
{json.dumps(evaluations, indent=2)}

Provide comprehensive feedback in JSON format:
{{
    "summary": "<2-3 sentence overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "areas_for_improvement": ["<area 1>", "<area 2>", "<area 3>"],
    "recommendations": ["<specific learning recommendation 1>", "<recommendation 2>"]
}}

Be encouraging but honest. Provide specific, actionable recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical interviewer providing constructive feedback. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=800,
            )

            assessment = json.loads(response.choices[0].message.content)
            return assessment

        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            # Fallback assessment
            return {
                "summary": f"You scored {overall_score}/100 on this {interview_data['skill']} interview.",
                "strengths": ["Completed the interview"],
                "areas_for_improvement": [
                    "Review the technical concepts covered",
                    "Practice explaining answers more clearly",
                ],
                "recommendations": [
                    f"Study {interview_data['skill']} fundamentals",
                    "Practice more technical interviews",
                ],
            }

    def format_report_for_display(self, report: Dict) -> str:
        """
        Format the report for text display (for voice readout or console).

        Args:
            report: The generated report dictionary

        Returns:
            Formatted string for display
        """
        output = []
        output.append("=" * 50)
        output.append("INTERVIEW FEEDBACK REPORT")
        output.append("=" * 50)
        output.append("")
        output.append(
            f"Category: {report['interview_metadata']['category'].title()}"
        )
        output.append(f"Skill: {report['interview_metadata']['skill'].title()}")
        output.append(f"Score: {report['overall_score']}/100")
        output.append("")
        output.append("SUMMARY:")
        output.append(report["summary"])
        output.append("")
        output.append("STRENGTHS:")
        for i, strength in enumerate(report["strengths"], 1):
            output.append(f"  {i}. {strength}")
        output.append("")
        output.append("AREAS FOR IMPROVEMENT:")
        for i, area in enumerate(report["areas_for_improvement"], 1):
            output.append(f"  {i}. {area}")
        output.append("")
        output.append("RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            output.append(f"  {i}. {rec}")
        output.append("")
        output.append("DETAILED QUESTION FEEDBACK:")
        for i, eval in enumerate(report["question_evaluations"], 1):
            output.append(f"\nQuestion {i}: {eval['question']}")
            output.append(f"Score: {eval['score']}/100")
            output.append(f"Feedback: {eval['feedback']}")
        output.append("")
        output.append("=" * 50)

        return "\n".join(output)