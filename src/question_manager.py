import pandas as pd
import random
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("question_manager")


class QuestionManager:
    """
    Manages interview questions loaded in memory for fast, low-latency access.
    
    Loads questions from Excel on initialization and keeps them in memory
    to avoid database calls during the interview (critical for voice latency).
    """

    def __init__(self, excel_path: str):
        """
        Initialize and load questions from Excel file into memory.

        Args:
            excel_path: Path to the Excel file containing questions
                       Expected columns: category, skill, question, answer
        """
        logger.info(f"Loading questions from {excel_path}...")
        
        try:
            # Load Excel file
            self.df = pd.read_excel(excel_path)
            
            # Validate required columns
            required_columns = ["category", "skill", "question", "answer"]
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean data
            self.df = self.df.dropna(subset=required_columns)
            
            # Normalize category and skill names (lowercase, strip whitespace)
            self.df["category"] = self.df["category"].str.lower().str.strip()
            self.df["skill"] = self.df["skill"].str.lower().str.strip()
            
            # Create index for fast lookup
            self._create_index()
            
            logger.info(f"✓ Loaded {len(self.df)} questions successfully")
            logger.info(f"✓ Categories: {len(self.df['category'].unique())}")
            logger.info(f"✓ Skills: {len(self.df['skill'].unique())}")
            
        except FileNotFoundError:
            logger.error(f"Excel file not found: {excel_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise

    def _create_index(self):
        """Create an index for fast question lookup by category and skill."""
        self.index = {}
        
        for category in self.df["category"].unique():
            self.index[category] = {}
            category_df = self.df[self.df["category"] == category]
            
            for skill in category_df["skill"].unique():
                skill_df = category_df[category_df["skill"] == skill]
                self.index[category][skill] = skill_df.index.tolist()

    def get_questions(
        self,
        category: str,
        skill: str,
        count: int = 5,
        random_selection: bool = True,
    ) -> List[Dict]:
        """
        Get questions for a specific category and skill.

        Args:
            category: Technology category (e.g., 'frontend', 'backend')
            skill: Specific skill (e.g., 'HTML', 'React')
            count: Number of questions to return (default: 5)
            random_selection: If True, randomly select questions; if False, return first N

        Returns:
            List of question dictionaries with keys: question, answer, category, skill
        """
        category = category.lower().strip()
        skill = skill.lower().strip()

        # Check if category and skill exist
        if category not in self.index:
            logger.warning(f"Category not found: {category}")
            return []

        if skill not in self.index[category]:
            logger.warning(f"Skill not found: {skill} in category {category}")
            return []

        # Get question indices
        question_indices = self.index[category][skill]

        if not question_indices:
            return []

        # Select questions
        if random_selection:
            selected_indices = random.sample(
                question_indices, min(count, len(question_indices))
            )
        else:
            selected_indices = question_indices[:count]

        # Get questions
        questions = []
        for idx in selected_indices:
            row = self.df.loc[idx]
            questions.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "skill": row["skill"],
                }
            )

        logger.info(f"Retrieved {len(questions)} questions for {category}/{skill}")
        return questions

    def get_available_categories(self) -> Dict[str, List[str]]:
        """
        Get all available categories and their skills.

        Returns:
            Dictionary mapping categories to lists of skills
        """
        categories = {}
        for category in self.df["category"].unique():
            skills = self.df[self.df["category"] == category]["skill"].unique().tolist()
            categories[category] = sorted(skills)

        return categories

    def search_questions(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        Search questions by keyword.

        Args:
            keyword: Keyword to search in questions
            limit: Maximum number of results

        Returns:
            List of matching questions
        """
        keyword = keyword.lower()
        mask = self.df["question"].str.lower().str.contains(keyword, na=False)
        results_df = self.df[mask].head(limit)

        questions = []
        for _, row in results_df.iterrows():
            questions.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "skill": row["skill"],
                }
            )

        return questions

    def get_stats(self) -> Dict:
        """Get statistics about the question database."""
        return {
            "total_questions": len(self.df),
            "categories": len(self.df["category"].unique()),
            "skills": len(self.df["skill"].unique()),
            "category_breakdown": self.df["category"].value_counts().to_dict(),
            "skill_breakdown": self.df["skill"].value_counts().to_dict(),
        }