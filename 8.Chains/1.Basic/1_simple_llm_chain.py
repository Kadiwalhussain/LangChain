"""
Basic Chain #1: Simple LLMChain
The most basic chain - connects a prompt template with an LLM
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

print("=" * 80)
print("SIMPLE LLM CHAIN")
print("=" * 80)

# Initialize LLM
llm = OllamaLLM(model="mistral")

# Create prompt template
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
question = "What is the capital of France?"
result = chain.invoke({"question": question})

print(f"\n📝 Input:")
print(f"   Question: {question}")

print(f"\n✅ Output:")
print(f"   {result['text']}")

# Multiple questions
print(f"\n" + "=" * 80)
print("RUNNING MULTIPLE INPUTS")
print("=" * 80)

questions = [
    "What is 2+2?",
    "Who invented Python?",
    "What is AI?"
]

for q in questions:
    result = chain.invoke({"question": q})
    print(f"\nQ: {q}")
    print(f"A: {result['text'][:100]}...")

print("\n" + "=" * 80)
print("CHAIN FLOW")
print("=" * 80)
print("""
Input Dict → Prompt Template → LLM → Output Dict
{"question": "..."} → "Question: ...\nAnswer:" → LLM → {"text": "..."}

Components:
1. PromptTemplate: Defines input structure
2. LLM: Generates response
3. Chain: Connects them together
""")
print("=" * 80)
