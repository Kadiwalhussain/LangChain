"""
Basic Output Parser #2: Comma Separated List Parser
Parses comma-separated values into a Python list
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Initialize LLM and parser
llm = OllamaLLM(model="mistral")
parser = CommaSeparatedListOutputParser()

print("=" * 80)
print("COMMA SEPARATED LIST OUTPUT PARSER")
print("=" * 80)

# Get format instructions
format_instructions = parser.get_format_instructions()
print(f"\n📋 Format Instructions:")
print(f"   {format_instructions}")

# Create prompt with format instructions
prompt = f"""List 5 programming languages.
{format_instructions}

Output only the comma-separated list, nothing else.
"""

print(f"\n💬 Prompt:")
print(f"   {prompt}")

# Chain LLM with parser
chain = llm | parser
result = chain.invoke(prompt)

print(f"\n✅ Parsed Result:")
print(f"   Type: {type(result)}")
print(f"   Value: {result}")

# Access individual items
print(f"\n📊 Individual Items:")
for i, item in enumerate(result, 1):
    print(f"   {i}. {item}")

print("\n" + "=" * 80)
print("FLOW DIAGRAM")
print("=" * 80)
print("""
Input → LLM → CommaSeparatedListOutputParser → List

Example:
"List 3 colors" → LLM → "red, green, blue" → ["red", "green", "blue"]
""")
print("=" * 80)
