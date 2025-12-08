"""
Basic Structured Output - Simple String Response
This is the most basic form of structured output where we get plain text from LLM
and return it as is.
"""

from langchain_ollama import OllamaLLM

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Simple prompt
prompt = "What is the capital of France?"

# Get response (this is already structured - it's just a string)
response = llm.invoke(prompt)

print("=" * 60)
print("BASIC STRING OUTPUT")
print("=" * 60)
print(f"Question: {prompt}")
print(f"Answer: {response}")
print("=" * 60)
