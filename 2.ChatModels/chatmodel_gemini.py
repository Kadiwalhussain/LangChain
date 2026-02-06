from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv  
import os
load_dotenv()


model = ChatGoogleGenAI(
    model="gemini-1.5-pro",
    temperature=1.5,)

result = model.invoke("what is the capital of india?")
print(result.content)

