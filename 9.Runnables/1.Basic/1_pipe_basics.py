"""
Runnable Basic #1: Pipe Operator Basics
Understanding the | operator and basic composition
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("=" * 80)
print("PIPE OPERATOR BASICS")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Method 1: Without pipe (old way)
print("\n1️⃣  Without Pipe Operator (Verbose):")
print("-" * 80)

template = "Tell me a fact about {topic}"
prompt = PromptTemplate.from_template(template)

# Manual chaining
formatted_prompt = prompt.invoke({"topic": "Python"})
print(f"Step 1 - Formatted Prompt: {formatted_prompt}")

llm_output = llm.invoke(formatted_prompt)
print(f"Step 2 - LLM Output: {llm_output[:100]}...")

parsed_output = parser.invoke(llm_output)
print(f"Step 3 - Parsed Output: {parsed_output[:100]}...")

# Method 2: With pipe (new way)
print("\n\n2️⃣  With Pipe Operator (Clean):")
print("-" * 80)

# Chain with pipe operator
chain = prompt | llm | parser
result = chain.invoke({"topic": "JavaScript"})

print(f"Result: {result[:150]}...")

# Method 3: Multiple topics
print("\n\n3️⃣  Processing Multiple Inputs:")
print("-" * 80)

topics = ["AI", "Blockchain", "Cloud Computing"]

for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"\n{topic}: {result[:100]}...")

print("\n" + "=" * 80)
print("PIPE OPERATOR EXPLAINED")
print("=" * 80)
print("""
The Pipe Operator |:

Before (Verbose):
    formatted = prompt.invoke(input)
    llm_output = llm.invoke(formatted)
    result = parser.invoke(llm_output)

After (Clean):
    chain = prompt | llm | parser
    result = chain.invoke(input)

Benefits:
✅ More readable
✅ Less code
✅ Easy to modify
✅ Functional composition
✅ Left-to-right flow

How it works:
Input → Component1 | Component2 | Component3 → Output
    Each | passes output to next component
""")
print("=" * 80)
