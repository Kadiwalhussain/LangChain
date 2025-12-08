from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# Using Ollama for open source LLM (requires Ollama to be running locally)
# Download models from https://ollama.ai
llm = Ollama(model="mistral")

result = llm.invoke("What is the capital of India?")
print(result)
