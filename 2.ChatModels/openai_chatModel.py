from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=1.5,
    max_tokens=50
)

result = model.invoke([
    {"role": "user", "content": "Write a 5-line poem on cricket"}
])

print(result.content)
