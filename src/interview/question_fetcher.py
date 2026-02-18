"""Fetches interview questions from ChromaDB. Called once per interview."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions


def _default_db_path() -> str:
    # Project root: src/interview/question_fetcher.py -> agent/interview_db
    return str(Path(__file__).resolve().parent.parent.parent / "interview_db")


class QuestionFetcher:
    """
    Fetches interview questions from ChromaDB.
    Called ONCE per interview; results are stored in LangGraph state.
    """

    def __init__(self, db_path: str | None = None) -> None:
        db_path = db_path or _default_db_path()
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(
            "interview_questions",
            embedding_function=embedding_fn,
        )

    def fetch_questions(
        self,
        category: str,
        skill: str,
        count: int = 10,
        level: str = "easy",
    ) -> list[dict[str, Any]]:
        """
        Fetch questions with metadata filtering first, then semantic ranking.
        Latency: ~20-50ms for 30k questions.
        Questions are unique; no repeats during the interview.
        """
        category = category.lower().strip()
        skill = skill.lower().strip()
        level = level.lower().strip()

        results = self.collection.query(
            query_texts=[f"{skill} {level} interview technical question"],
            n_results=count,
            where={
                "$and": [
                    {"category": {"$eq": category}},
                    {"skill": {"$eq": skill}},
                    {"level": {"$eq": level}},
                ]
            },
        )

        if not results["documents"] or not results["documents"][0]:
            results = self.collection.query(
                query_texts=[f"{skill} interview question"],
                n_results=count,
                where={
                    "$and": [
                        {"category": {"$eq": category}},
                        {"skill": {"$eq": skill}},
                    ]
                },
            )

        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []

        questions: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for doc, meta in zip(documents, metadatas):
            qid = meta.get("question_id", "")
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            questions.append(
                {
                    "question": doc,
                    "answer": meta.get("answer", ""),
                    "answer_keywords": meta.get("answer_keywords", ""),
                    "level": meta.get("level", "easy"),
                    "question_id": qid,
                }
            )
            if len(questions) >= count:
                break

        return questions
