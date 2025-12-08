"""
Basic Structured Output - Key-Value Pairs
Getting data as key-value format
"""

from langchain_ollama import OllamaLLM

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Prompt that asks for key-value output
prompt = """
Provide information about Python programming language in key-value format.
Return ONLY key-value pairs (no markdown, no extra text):

Name: Python
Type: Programming Language
Year Created: 1991
Creator: Guido van Rossum
"""

# Get response
response = llm.invoke(prompt)

print("=" * 60)
print("KEY-VALUE STRUCTURED OUTPUT")
print("=" * 60)

# Parse key-value pairs
data = {}
lines = response.strip().split('\n')

print("\n🔑 Data:")
for line in lines:
    line = line.strip()
    if ':' in line:
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        data[key] = value
        print(f"  {key}: {value}")

print(f"\n✅ Total properties: {len(data)}")

# Access specific values
print("\n📌 Accessing specific values:")
if 'Name' in data:
    print(f"  Language name: {data['Name']}")
if 'Year Created' in data:
    print(f"  Creation year: {data['Year Created']}")

print("=" * 60)
