"""
Basic Chain #3: Transform Chain
Apply custom transformations to input/output
"""

from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

print("=" * 80)
print("TRANSFORM CHAIN")
print("=" * 80)

# Define a transform function
def transform_text(inputs: dict) -> dict:
    """Transform function: uppercase the text and count words"""
    text = inputs["text"]
    transformed = {
        "uppercase_text": text.upper(),
        "word_count": len(text.split()),
        "original": text
    }
    return transformed

# Create transform chain
transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["uppercase_text", "word_count", "original"],
    transform=transform_text
)

# Test the transform chain
input_text = "hello world this is a test"
print(f"\n📝 Input:")
print(f"   {input_text}")

result = transform_chain.invoke({"text": input_text})

print(f"\n✅ Transformed Output:")
print(f"   Uppercase: {result['uppercase_text']}")
print(f"   Word Count: {result['word_count']}")
print(f"   Original: {result['original']}")

# Now combine with LLM
print(f"\n" + "=" * 80)
print("TRANSFORM + LLM CHAIN")
print("=" * 80)

llm = OllamaLLM(model="mistral")

# Create LLM chain that uses transformed data
template = """Analyze this text:
Original: {original}
Uppercase: {uppercase_text}
Word Count: {word_count}

Provide a brief analysis:"""

prompt = PromptTemplate(
    input_variables=["original", "uppercase_text", "word_count"],
    template=template
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Combine transform and LLM chains
combined_chain = SequentialChain(
    chains=[transform_chain, llm_chain],
    input_variables=["text"],
    output_variables=["text"],
    verbose=True
)

result = combined_chain.invoke({"text": "Python is a powerful programming language"})

print(f"\n✅ Final Analysis:")
print(f"   {result['text']}")

print("\n" + "=" * 80)
print("TRANSFORM CHAIN USE CASES")
print("=" * 80)
print("""
Use TransformChain when you need to:

✅ Preprocess input (clean, format, validate)
✅ Extract specific information
✅ Calculate metrics
✅ Convert formats
✅ Apply custom logic

Examples:
- Extract URLs from text
- Count tokens before sending to LLM
- Clean/sanitize user input
- Parse structured data
- Calculate statistics

Flow:
Input → Transform Function → Transformed Data → Next Chain
""")
print("=" * 80)
