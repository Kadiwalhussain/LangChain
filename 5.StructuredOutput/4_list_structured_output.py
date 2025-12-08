"""
Basic Structured Output - List/Array Format
Getting multiple items in a structured list format
"""

from langchain_ollama import OllamaLLM

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Prompt that asks for list output
prompt = """
List 5 benefits of machine learning. 
Return ONLY a numbered list (no other text):

1. Benefit one
2. Benefit two
"""

# Get response
response = llm.invoke(prompt)

print("=" * 60)
print("LIST STRUCTURED OUTPUT")
print("=" * 60)

# Parse list
lines = response.strip().split('\n')

print("\n📝 Items:")
items = []
for line in lines:
    line = line.strip()
    if line:  # Skip empty lines
        # Remove numbering
        cleaned = line
        if line[0].isdigit() and '.' in line:
            cleaned = line.split('.', 1)[1].strip()
        
        items.append(cleaned)
        print(f"  • {cleaned}")

print(f"\n✅ Total items: {len(items)}")
print("=" * 60)
