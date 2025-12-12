"""
Intermediate Output Parser #2: Structured Output Parser
Define output schema without Pydantic
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

# Define schema using ResponseSchema
response_schemas = [
    ResponseSchema(name="product_name", description="Name of the product"),
    ResponseSchema(name="price", description="Price in USD"),
    ResponseSchema(name="category", description="Product category"),
    ResponseSchema(name="in_stock", description="Whether product is in stock (true/false)")
]

# Initialize LLM and parser
llm = OllamaLLM(model="mistral")
parser = StructuredOutputParser.from_response_schemas(response_schemas)

print("=" * 80)
print("STRUCTURED OUTPUT PARSER")
print("=" * 80)

# Get format instructions
format_instructions = parser.get_format_instructions()
print(f"\n📋 Format Instructions:")
print(format_instructions)

# Create prompt
prompt = f"""Generate information about a laptop product.

{format_instructions}
"""

print(f"\n💬 Prompt sent to LLM")

# Chain LLM with parser
chain = llm | parser

try:
    result = chain.invoke(prompt)
    
    print(f"\n✅ Parsed Structured Output:")
    print(f"   Type: {type(result)}")
    print(f"   Product: {result.get('product_name')}")
    print(f"   Price: ${result.get('price')}")
    print(f"   Category: {result.get('category')}")
    print(f"   In Stock: {result.get('in_stock')}")
    
    print(f"\n📊 Full Result:")
    print(f"   {result}")
    
except Exception as e:
    print(f"\n❌ Parsing failed: {e}")

print("\n" + "=" * 80)
print("WHEN TO USE")
print("=" * 80)
print("""
Use StructuredOutputParser when:
✅ You don't want to create Pydantic models
✅ Simple schema definition
✅ Quick prototyping
✅ Don't need validation beyond basic structure

Use PydanticOutputParser when:
✅ Need complex validation
✅ Want type safety
✅ Need nested models
✅ Production applications
""")
print("=" * 80)
