"""
Intermediate Output Parser #1: Pydantic Parser
Use Pydantic models to parse and validate LLM output
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define Pydantic model
class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")
    city: str = Field(description="City where person lives")

# Initialize LLM and parser
llm = OllamaLLM(model="mistral")
parser = PydanticOutputParser(pydantic_object=Person)

print("=" * 80)
print("PYDANTIC OUTPUT PARSER")
print("=" * 80)

# Get format instructions
format_instructions = parser.get_format_instructions()
print(f"\n📋 Format Instructions from Parser:")
print(format_instructions)

# Create prompt
prompt = f"""Generate information about a software engineer.

{format_instructions}

Return ONLY the JSON, no other text.
"""

print(f"\n💬 Full Prompt:")
print(prompt)

# Chain LLM with parser
chain = llm | parser

try:
    result = chain.invoke(prompt)
    
    print(f"\n✅ Parsed Pydantic Model:")
    print(f"   Type: {type(result)}")
    print(f"   Name: {result.name}")
    print(f"   Age: {result.age}")
    print(f"   Occupation: {result.occupation}")
    print(f"   City: {result.city}")
    
    print(f"\n📊 As Dictionary:")
    print(f"   {result.model_dump()}")
    
    print(f"\n📝 As JSON:")
    print(f"   {result.model_dump_json(indent=2)}")
    
except Exception as e:
    print(f"\n❌ Parsing/Validation failed: {e}")

print("\n" + "=" * 80)
print("FLOW DIAGRAM")
print("=" * 80)
print("""
Input → LLM → PydanticOutputParser → Pydantic Model (validated)

Example:
"Generate person info" → LLM → '{"name":"John","age":30,...}' 
  → Person(name="John", age=30, ...)
  
Benefits:
✅ Type validation
✅ Field validation  
✅ Clear data structure
✅ IDE autocomplete
""")
print("=" * 80)
