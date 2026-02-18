"""
Build ChromaDB index from Excel or CSV. Run once before using the interview agent.

Usage:
    uv run python build_index.py [path/to/questions.xlsx] [--db-path ./interview_db]
    uv run python build_index.py path/to/questions.csv

Creates ./interview_db by default. Idempotent — safe to re-run.
"""

import argparse
import contextlib
import hashlib
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions


def _default_db_path() -> Path:
    return Path(__file__).resolve().parent / "interview_db"


def _load_questions(path: str | Path) -> pd.DataFrame:
    """Load questions from Excel (.xlsx, .xls) or CSV (.csv)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported format: {suffix}. Use .csv, .xlsx, or .xls"
    )


def build_interview_index(
    input_path: str,
    db_path: str | Path | None = None,
) -> str:
    """
    Build ChromaDB index from Excel or CSV file.
    Expects columns: category, skill, question, answer; optional: level, answer_keywords, topic_tags.
    """
    db_path = Path(db_path or _default_db_path())
    db_path = db_path.resolve()

    df = _load_questions(input_path)
    required = ["category", "skill", "question", "answer"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"File must have column: {col}. Found: {list(df.columns)}"
            )
    df = df.dropna(subset=required)
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["skill"] = df["skill"].astype(str).str.lower().str.strip()

    if "level" not in df.columns and "difficulty" in df.columns:
        df["level"] = df["difficulty"].astype(str).str.lower().str.strip()
    elif "level" not in df.columns:
        df["level"] = "easy"
    else:
        df["level"] = df["level"].astype(str).str.lower().str.strip()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(db_path))

    with contextlib.suppress(Exception):
        client.delete_collection("interview_questions")

    collection = client.create_collection(
        name="interview_questions",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []
    batch_size = 1000

    # Normalize skill for IDs (ChromaDB ids must be unique; CSV may have duplicate questions)
    def _safe(s: str) -> str:
        return s.replace(" ", "_").lower()

    for idx, (_, row) in enumerate(df.iterrows()):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        category = str(row["category"]).strip()
        skill = str(row["skill"]).strip()
        level = str(row["level"]).strip() or "easy"

        uid = hashlib.md5(f"{category}_{skill}_{question}_{idx}".encode()).hexdigest()[:12]
        doc_id = f"{_safe(category)}_{_safe(skill)}_{uid}"

        documents.append(question)
        ids.append(doc_id)
        meta = {
            "category": category,
            "skill": skill,
            "level": level,
            "answer": answer[:500],
            "question_id": doc_id,
        }
        if "answer_keywords" in df.columns and pd.notna(row.get("answer_keywords")):
            meta["answer_keywords"] = str(row["answer_keywords"]).strip()[:500]
        if "topic_tags" in df.columns and pd.notna(row.get("topic_tags")):
            meta["topic_tags"] = str(row["topic_tags"]).strip()[:500]
        metadatas.append(meta)

    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"Indexed {end}/{len(documents)}")

    print(f"\nDone. Total: {len(documents)} questions indexed.")
    print(f"DB saved to: {db_path}")
    return str(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB index from Excel or CSV"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="questions.xlsx",
        help="Path to questions file: .csv, .xlsx, or .xls (default: questions.xlsx)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="ChromaDB output path (default: ./interview_db)",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")
    build_interview_index(str(input_path), args.db_path)


if __name__ == "__main__":
    main()
