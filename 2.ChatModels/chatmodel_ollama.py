from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Using ChatOllama for open source chat models (requires Ollama running)
# Popular models: mistral, llama2, neural-chat, etc.
model = ChatOllama(
    model="mistral",
    temperature=0.7
)

result = model.invoke("What is the capital of India?")
print(result.content)
