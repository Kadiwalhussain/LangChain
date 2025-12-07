from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv


load_dotenv()


model = ChatAnthropic(model="claude-2", temperature=1.5, max_tokens=50)

result = model.invoke("what is the capital of india?")

print(result.content)
