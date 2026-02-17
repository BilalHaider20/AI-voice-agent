"""
Script to create a sample questions Excel file with the correct format.
This is a template - replace with your actual 30,000+ questions.
"""

import pandas as pd

# Sample data - replace this with your actual questions
sample_questions = [
    {
        "category": "frontend",
        "skill": "html",
        "question": "What are inline elements?",
        "answer": "They do not occupy new line and do not break the flow",
    },
    {
        "category": "frontend",
        "skill": "html",
        "question": "What is the difference between div and span?",
        "answer": "Div is a block-level element while span is an inline element",
    },
    {
        "category": "frontend",
        "skill": "html",
        "question": "What is semantic HTML?",
        "answer": "HTML that introduces meaning to the web page rather than just presentation",
    },
    {
        "category": "frontend",
        "skill": "css",
        "question": "What is the box model?",
        "answer": "The CSS box model consists of content, padding, border, and margin",
    },
    {
        "category": "frontend",
        "skill": "css",
        "question": "What is flexbox?",
        "answer": "A one-dimensional layout method for arranging items in rows or columns",
    },
    {
        "category": "frontend",
        "skill": "javascript",
        "question": "What is a closure?",
        "answer": "A function that has access to variables in its outer lexical scope",
    },
    {
        "category": "frontend",
        "skill": "javascript",
        "question": "What is the difference between let and var?",
        "answer": "Let is block-scoped while var is function-scoped",
    },
    {
        "category": "frontend",
        "skill": "react",
        "question": "What are React hooks?",
        "answer": "Functions that let you use state and other React features without writing a class",
    },
    {
        "category": "backend",
        "skill": "python",
        "question": "What is a decorator?",
        "answer": "A function that takes another function and extends its behavior without explicitly modifying it",
    },
    {
        "category": "backend",
        "skill": "python",
        "question": "What is the difference between list and tuple?",
        "answer": "Lists are mutable while tuples are immutable",
    },
    {
        "category": "backend",
        "skill": "nodejs",
        "question": "What is the event loop?",
        "answer": "A mechanism that handles asynchronous operations in Node.js by executing callbacks when operations complete",
    },
    {
        "category": "database",
        "skill": "sql",
        "question": "What is a primary key?",
        "answer": "A unique identifier for a record in a database table",
    },
    {
        "category": "database",
        "skill": "sql",
        "question": "What is normalization?",
        "answer": "The process of organizing data to minimize redundancy and improve data integrity",
    },
    {
        "category": "database",
        "skill": "mongodb",
        "question": "What is a document in MongoDB?",
        "answer": "A record in MongoDB stored in BSON format, similar to JSON objects",
    },
]

# Create DataFrame
df = pd.DataFrame(sample_questions)

# Save to Excel
output_file = "questions.xlsx"
df.to_excel(output_file, index=False)

print(f"✓ Sample questions file created: {output_file}")
print(f"✓ Total questions: {len(df)}")
print(f"✓ Categories: {df['category'].unique().tolist()}")
print(f"✓ Skills: {df['skill'].unique().tolist()}")
print("\nNow replace this file with your 30,000+ questions dataset!")
print("Required columns: category, skill, question, answer")