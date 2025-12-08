from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

# Using HuggingFace models via HuggingFaceHub API
# Get your token from https://huggingface.co/settings/tokens
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_length": 128}
)

result = llm.invoke("What is the capital of India?")
print(result)
