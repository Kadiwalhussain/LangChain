"""
Intermediate Output Parser #3: XML Parser
Parse XML output from LLM
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import XMLOutputParser

# Initialize LLM and parser
llm = OllamaLLM(model="mistral")
parser = XMLOutputParser()

print("=" * 80)
print("XML OUTPUT PARSER")
print("=" * 80)

# Create prompt asking for XML
prompt = """Generate information about a book in XML format.
Return ONLY valid XML:

<book>
    <title>Book Title</title>
    <author>Author Name</author>
    <year>2024</year>
    <genre>Fiction</genre>
</book>

Generate a book about Python programming. Return ONLY the XML.
"""

print(f"\n💬 Prompt: (asking for XML format)")

# Chain LLM with parser
chain = llm | parser

try:
    result = chain.invoke(prompt)
    
    print(f"\n✅ Parsed XML:")
    print(f"   Type: {type(result)}")
    print(f"   Title: {result.get('book', {}).get('title')}")
    print(f"   Author: {result.get('book', {}).get('author')}")
    print(f"   Year: {result.get('book', {}).get('year')}")
    print(f"   Genre: {result.get('book', {}).get('genre')}")
    
    print(f"\n📊 Full Parsed Structure:")
    import json
    print(json.dumps(result, indent=2))
    
except Exception as e:
    print(f"\n❌ Parsing failed: {e}")
    print(f"   LLM might have returned invalid XML")

print("\n" + "=" * 80)
print("XML vs JSON")
print("=" * 80)
print("""
XML:
✅ Better for document-like structures
✅ Attributes and nested elements
✅ Schema validation with XSD
❌ More verbose
❌ Harder for LLMs to generate correctly

JSON:
✅ Simpler syntax
✅ Native JavaScript support
✅ Easier for LLMs
✅ Less verbose
❌ Limited metadata
""")
print("=" * 80)
