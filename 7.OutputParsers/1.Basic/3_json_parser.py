"""
Basic Output Parser #3: Simple JSON Parser
Parses JSON output from LLM
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM and parser
llm = OllamaLLM(model="mistral")
parser = JsonOutputParser()

print("=" * 80)
print("JSON OUTPUT PARSER")
print("=" * 80)

# Create prompt asking for JSON
prompt = """Provide information about Python in JSON format.
Return ONLY valid JSON with these fields:
- name: The name of the language
- year: Year it was created
- creator: Who created it

Example format:
{
  "name": "Python",
  "year": 1991,
  "creator": "Guido van Rossum"
}

Return ONLY the JSON, no other text.
"""

print(f"\n💬 Prompt: (asking for JSON)")

# Chain LLM with parser
chain = llm | parser

try:
    result = chain.invoke(prompt)
    
    print(f"\n✅ Parsed JSON:")
    print(f"   Type: {type(result)}")
    print(f"   Value: {result}")
    
    print(f"\n📊 Accessing Fields:")
    print(f"   Name: {result.get('name')}")
    print(f"   Year: {result.get('year')}")
    print(f"   Creator: {result.get('creator')}")
    
except Exception as e:
    print(f"\n❌ Parsing failed: {e}")
    print(f"   LLM might have returned invalid JSON")

print("\n" + "=" * 80)
print("FLOW DIAGRAM")
print("=" * 80)
print("""
Input → LLM → JsonOutputParser → Dictionary

Example:
"Info about Python" → LLM → '{"name":"Python","year":1991}' → {"name": "Python", "year": 1991}
""")
print("=" * 80)
