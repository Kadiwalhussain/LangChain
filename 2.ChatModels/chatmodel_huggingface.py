from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Using HuggingFace models as chat models
# Get your token from https://huggingface.co/settings/tokens
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    max_length=128,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)
