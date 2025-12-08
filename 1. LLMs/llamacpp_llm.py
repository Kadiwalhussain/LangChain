from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv

load_dotenv()

# Using LlamaCpp for running quantized models locally
# Download GGUF models from https://huggingface.co/models?library=gguf
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=512,
    top_p=1,
    verbose=False
)

result = llm.invoke("What is the capital of India?")
print(result)
