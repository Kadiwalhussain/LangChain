import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Verify the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Create LLM object
llm = ChatOpenAI(
    model="gpt-4o-mini",   # change model if needed
    api_key=api_key
)

# Invoke LLM
response = llm.invoke("What is the capital of India?")

print(response.content)
