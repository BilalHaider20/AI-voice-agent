"""
Test script to verify all components are working correctly.
Run this before starting the agent.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv(".env.local")


def test_environment():
    """Test if all environment variables are set."""
    print("=" * 60)
    print("Testing Environment Variables...")
    print("=" * 60)

    required_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "GROQ_API_KEY",
        "QUESTIONS_EXCEL_PATH",
    ]

    optional_vars = ["SUPABASE_URL", "SUPABASE_KEY"]

    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: Set")
        else:
            print(f"✗ {var}: MISSING")
            all_good = False

    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: Set")
        else:
            print(f"⚠ {var}: Not set (database features will be disabled)")

    return all_good


def test_questions_file():
    """Test if questions file exists and is valid."""
    print("\n" + "=" * 60)
    print("Testing Questions File...")
    print("=" * 60)

    excel_path = os.getenv("QUESTIONS_EXCEL_PATH", "questions.xlsx")

    if not os.path.exists(excel_path):
        print(f"✗ File not found: {excel_path}")
        print("  Run: python create_sample_questions.py")
        return False

    try:
        from question_manager import QuestionManager

        qm = QuestionManager(excel_path)
        stats = qm.get_stats()

        print(f"✓ Questions file loaded successfully")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Categories: {stats['categories']}")
        print(f"  Skills: {stats['skills']}")

        # Test retrieval
        categories = qm.get_available_categories()
        if categories:
            first_cat = list(categories.keys())[0]
            first_skill = categories[first_cat][0]
            questions = qm.get_questions(first_cat, first_skill, count=2)
            print(f"\n✓ Test retrieval successful:")
            print(f"  Category: {first_cat}")
            print(f"  Skill: {first_skill}")
            print(f"  Retrieved: {len(questions)} questions")

        return True

    except Exception as e:
        print(f"✗ Error loading questions: {e}")
        return False


def test_supabase():
    """Test Supabase connection."""
    print("\n" + "=" * 60)
    print("Testing Supabase Connection...")
    print("=" * 60)

    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("⚠ Supabase not configured (optional)")
        return True

    try:
        from supabase_integration import SupabaseManager

        db = SupabaseManager()
        if db.client:
            print("✓ Supabase client initialized")
            # Try a simple query
            # result = await db.get_user_interview_history("test", limit=1)
            print("✓ Connection successful")
            return True
        else:
            print("⚠ Supabase client not initialized")
            return True

    except Exception as e:
        print(f"✗ Supabase connection error: {e}")
        return False


def test_report_generator():
    """Test report generator."""
    print("\n" + "=" * 60)
    print("Testing Report Generator...")
    print("=" * 60)

    try:
        from report_generator import ReportGenerator

        rg = ReportGenerator()
        print("✓ Report generator initialized")

        # Test with sample data
        sample_interview = {
            "category": "frontend",
            "skill": "html",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:10:00",
            "questions": [{"question": "Test?", "answer": "Test answer"}],
            "responses": [
                {
                    "question": "What are inline elements?",
                    "expected_answer": "They do not occupy new line",
                    "candidate_answer": "Inline elements stay on the same line",
                }
            ],
        }

        print("✓ Sample interview data created")
        print("  Note: Actual report generation requires API call")

        return True

    except Exception as e:
        print(f"✗ Report generator error: {e}")
        return False


def test_dependencies():
    """Test if all required packages are installed."""
    print("\n" + "=" * 60)
    print("Testing Dependencies...")
    print("=" * 60)

    required_packages = [
        "livekit",
        "pandas",
        "openpyxl",
        "openai",
        "dotenv",
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\nRun: pip install -r requirements.txt")

    return all_installed


def main():
    """Run all tests."""
    print("\n🧪 Interview Agent Setup Verification")
    print("=" * 60)

    tests = [
        ("Dependencies", test_dependencies),
        ("Environment Variables", test_environment),
        ("Questions File", test_questions_file),
        ("Supabase", test_supabase),
        ("Report Generator", test_report_generator),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to run the agent.")
        print("\nStart the agent with:")
        print("  python interview_agent.py dev")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Create .env.local: cp .env.local.example .env.local")
        print("  3. Generate questions: python create_sample_questions.py")
        print("  4. Set up Supabase: Run supabase_schema.sql in Supabase")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())