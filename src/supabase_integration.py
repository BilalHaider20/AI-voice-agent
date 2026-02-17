import os
from typing import Dict, Optional
from datetime import datetime
import logging
from supabase import create_client, Client

logger = logging.getLogger("supabase_integration")


class SupabaseManager:
    """
    Manages Supabase database operations for interview sessions and reports.
    
    Database Schema (create these tables in Supabase):
    
    1. interviews:
        - id (uuid, primary key)
        - user_id (text)
        - category (text)
        - skill (text)
        - start_time (timestamp)
        - end_time (timestamp)
        - status (text: 'in_progress', 'completed', 'abandoned')
        - created_at (timestamp)
    
    2. interview_responses:
        - id (uuid, primary key)
        - interview_id (uuid, foreign key -> interviews.id)
        - question_index (int)
        - question (text)
        - expected_answer (text)
        - candidate_answer (text)
        - score (numeric)
        - feedback (text)
        - created_at (timestamp)
    
    3. interview_reports:
        - id (uuid, primary key)
        - interview_id (uuid, foreign key -> interviews.id)
        - overall_score (numeric)
        - summary (text)
        - strengths (jsonb)
        - areas_for_improvement (jsonb)
        - recommendations (jsonb)
        - full_report (jsonb)
        - created_at (timestamp)
    """

    def __init__(self):
        """Initialize Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning(
                "Supabase credentials not found. Database operations will be disabled."
            )
            self.client = None
        else:
            self.client: Client = create_client(supabase_url, supabase_key)
            logger.info("✓ Supabase client initialized")

    async def create_interview_session(
        self, user_id: str, category: str, skill: str
    ) -> Optional[str]:
        """
        Create a new interview session in the database.

        Args:
            user_id: Identifier for the user
            category: Interview category
            skill: Skill being tested

        Returns:
            Interview ID (UUID) if successful, None otherwise
        """
        if not self.client:
            logger.warning("Supabase not configured, skipping database save")
            return None

        try:
            data = {
                "user_id": user_id,
                "category": category,
                "skill": skill,
                "start_time": datetime.now().isoformat(),
                "status": "in_progress",
            }

            result = self.client.table("interviews").insert(data).execute()

            interview_id = result.data[0]["id"]
            logger.info(f"✓ Created interview session: {interview_id}")
            return interview_id

        except Exception as e:
            logger.error(f"Error creating interview session: {e}")
            return None

    async def save_interview_response(
        self,
        interview_id: str,
        question_index: int,
        question: str,
        expected_answer: str,
        candidate_answer: str,
        score: float,
        feedback: str,
    ) -> bool:
        """
        Save an individual question response.

        Returns:
            True if successful, False otherwise
        """
        if not self.client or not interview_id:
            return False

        try:
            data = {
                "interview_id": interview_id,
                "question_index": question_index,
                "question": question,
                "expected_answer": expected_answer,
                "candidate_answer": candidate_answer,
                "score": score,
                "feedback": feedback,
            }

            self.client.table("interview_responses").insert(data).execute()
            logger.info(f"✓ Saved response for question {question_index}")
            return True

        except Exception as e:
            logger.error(f"Error saving interview response: {e}")
            return False

    async def save_interview_report(
        self, interview_id: str, report: Dict
    ) -> Optional[str]:
        """
        Save the final interview report.

        Args:
            interview_id: Interview session ID
            report: Generated report dictionary

        Returns:
            Report ID if successful, None otherwise
        """
        if not self.client or not interview_id:
            return None

        try:
            data = {
                "interview_id": interview_id,
                "overall_score": report["overall_score"],
                "summary": report["summary"],
                "strengths": report["strengths"],
                "areas_for_improvement": report["areas_for_improvement"],
                "recommendations": report["recommendations"],
                "full_report": report,
            }

            result = self.client.table("interview_reports").insert(data).execute()

            # Update interview status to completed
            self.client.table("interviews").update(
                {"status": "completed", "end_time": datetime.now().isoformat()}
            ).eq("id", interview_id).execute()

            report_id = result.data[0]["id"]
            logger.info(f"✓ Saved interview report: {report_id}")
            return report_id

        except Exception as e:
            logger.error(f"Error saving interview report: {e}")
            return None

    async def get_user_interview_history(
        self, user_id: str, limit: int = 10
    ) -> list:
        """
        Get a user's interview history.

        Args:
            user_id: User identifier
            limit: Maximum number of interviews to return

        Returns:
            List of interview records
        """
        if not self.client:
            return []

        try:
            result = (
                self.client.table("interviews")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data

        except Exception as e:
            logger.error(f"Error fetching interview history: {e}")
            return []

    async def get_interview_report(self, interview_id: str) -> Optional[Dict]:
        """
        Retrieve a specific interview report.

        Args:
            interview_id: Interview session ID

        Returns:
            Report dictionary if found, None otherwise
        """
        if not self.client:
            return None

        try:
            result = (
                self.client.table("interview_reports")
                .select("*")
                .eq("interview_id", interview_id)
                .single()
                .execute()
            )

            return result.data["full_report"]

        except Exception as e:
            logger.error(f"Error fetching interview report: {e}")
            return None