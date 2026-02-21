from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
    api_key=api_key,  
    model="llama-3.3-70b-versatile",            
    temperature=0,
)