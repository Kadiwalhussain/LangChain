"""
Advanced Output Parser #1: Retry Parser
Automatically retry parsing with error feedback
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
from pydantic import BaseModel, Field, field_validator

# Define strict Pydantic model with validation
class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD", gt=0)
    rating: float = Field(description="Rating from 0 to 5")
    
    @field_validator('rating')
    @classmethod
    def rating_must_be_valid(cls, v):
        if v < 0 or v > 5:
            raise ValueError('Rating must be between 0 and 5')
        return v

# Initialize
llm = OllamaLLM(model="mistral")
base_parser = PydanticOutputParser(pydantic_object=Product)

print("=" * 80)
print("RETRY WITH ERROR OUTPUT PARSER")
print("=" * 80)

# Create prompt that might produce invalid output
prompt = f"""Generate a product with the following details.

{base_parser.get_format_instructions()}

Make the rating 7 out of 10 (this will fail validation).
"""

print(f"\n💬 Prompt: (intentionally requesting invalid rating)")

# Try with basic parser (will fail)
print(f"\n1️⃣  Without Retry Parser:")
chain_basic = llm | base_parser

try:
    result = chain_basic.invoke(prompt)
    print(f"   ✅ Success: {result}")
except Exception as e:
    print(f"   ❌ Failed: {str(e)[:100]}...")

# Now with retry parser
print(f"\n2️⃣  With Retry Parser:")
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=llm
)

# For retry parser, we need the original completion
completion = llm.invoke(prompt)
print(f"   Initial LLM output: {completion[:100]}...")

try:
    result = retry_parser.parse_with_prompt(completion, prompt)
    print(f"\n   ✅ Success after retry!")
    print(f"   Name: {result.name}")
    print(f"   Price: ${result.price}")
    print(f"   Rating: {result.rating}/5")
except Exception as e:
    print(f"   ❌ Still failed: {e}")

print("\n" + "=" * 80)
print("HOW RETRY PARSER WORKS")
print("=" * 80)
print("""
1. LLM generates output
2. Parser tries to parse → Fails with error
3. RetryParser sends error back to LLM
4. LLM corrects the output
5. Parser tries again
6. Success! ✅

Flow:
Input → LLM → Invalid Output → Parser (Error)
                ↓
            Retry Parser
                ↓
    "Fix this error: ..." → LLM → Valid Output → Parser → Success

Benefits:
✅ Automatic error correction
✅ Better success rate
✅ Handles edge cases
❌ More LLM calls (costs)
❌ Slower
""")
print("=" * 80)
