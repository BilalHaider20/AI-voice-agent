import sqlite3
import pandas as pd
import json
from collections import defaultdict

DB_NAME = "interviewer.db"
MAP_NAME = "map.json"

def setup():
    # 1. Connect and Create Table
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Using WITHOUT ROWID for better UUID primary key performance
    # cursor.execute("""
    #     CREATE TABLE IF NOT EXISTS questions (
    #         id TEXT PRIMARY KEY,
    #         category TEXT,
    #         skill TEXT,
    #         level TEXT,
    #         question TEXT NOT NULL,
    #         answer TEXT,
    #         created_at TEXT
    #     ) WITHOUT ROWID;
    # """)
    
    # 2. Load your data (Assuming a CSV export from Supabase)
    # If using a different format, adjust this part
    # df = pd.read_csv("data/Frontend_Interview_Questions.csv")
    # df.to_sql("questions", conn, if_exists="replace", index=False)
    
    # 3. Create the Index for O(1) Fetching
    # cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_questions_id ON questions(id);")
    
    # 4. Generate the map.json hierarchy
    cursor.execute("SELECT * FROM questions")
    rows = cursor.fetchall()
    print(rows)
    
    # Nested dictionary: category -> skill -> level -> [ids]
    # hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # for q_id, cat, skill, lvl in rows:
    #     hierarchy[cat][skill][lvl].append(q_id)
        
    # with open(MAP_NAME, "w") as f:
    #     json.dump({"data": hierarchy}, f, indent=2)
    
    # conn.commit()
    conn.close()
    # print(f"✅ Success! {DB_NAME} and {MAP_NAME} are ready.")

if __name__ == "__main__":
    setup()