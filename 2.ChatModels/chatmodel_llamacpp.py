from langchain_community.chat_models import ChatLlamaCpp
from dotenv import load_dotenv

load_dotenv()

# Using ChatLlamaCpp for running quantized models locally
# Download GGUF models from https://huggingface.co/models?library=gguf
model = ChatLlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=512,
    verbose=False
)

result = model.invoke("Write a 5-line poem on cricket")
print(result.content)
