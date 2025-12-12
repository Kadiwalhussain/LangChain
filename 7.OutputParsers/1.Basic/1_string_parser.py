"""
Basic Output Parser #1: String Output Parser
The simplest parser - just returns the LLM output as a string
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = OllamaLLM(model="mistral")

# Create string parser
parser = StrOutputParser()

print("=" * 80)
print("STRING OUTPUT PARSER - BASIC")
print("=" * 80)

# Direct LLM call
response = llm.invoke("What is 2+2?")
print(f"\n1️⃣  Direct LLM Response:")
print(f"   Type: {type(response)}")
print(f"   Value: {response}")

# With parser
parsed_response = parser.invoke(response)
print(f"\n2️⃣  Parsed Response:")
print(f"   Type: {type(parsed_response)}")
print(f"   Value: {parsed_response}")

# Chain LLM with parser
chain = llm | parser
result = chain.invoke("What is the capital of France?")

print(f"\n3️⃣  Chained LLM + Parser:")
print(f"   Type: {type(result)}")
print(f"   Value: {result}")

print("\n" + "=" * 80)
print("FLOW DIAGRAM")
print("=" * 80)
print("""
Input → LLM → StrOutputParser → String Output

Example:
"What is 2+2?" → LLM → "2+2 equals 4" → "2+2 equals 4" (string)
""")
print("=" * 80)
