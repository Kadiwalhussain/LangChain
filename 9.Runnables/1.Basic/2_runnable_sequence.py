"""
Runnable Basic #2: RunnableSequence
Understanding how pipe creates RunnableSequence
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

print("=" * 80)
print("RUNNABLE SEQUENCE")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
prompt = PromptTemplate.from_template("Explain {concept} in simple terms:")
parser = StrOutputParser()

# Method 1: Using pipe (creates RunnableSequence automatically)
print("\n1️⃣  Creating RunnableSequence with Pipe:")
print("-" * 80)

chain = prompt | llm | parser
print(f"Type: {type(chain)}")
print(f"Chain: {chain}")

# Method 2: Creating RunnableSequence explicitly
print("\n2️⃣  Creating RunnableSequence Explicitly:")
print("-" * 80)

explicit_chain = RunnableSequence(first=prompt, middle=[llm], last=parser)
print(f"Type: {type(explicit_chain)}")

# Both work the same way
print("\n3️⃣  Testing Both Chains:")
print("-" * 80)

test_input = {"concept": "machine learning"}

result1 = chain.invoke(test_input)
print(f"\nPipe Result: {result1[:150]}...")

result2 = explicit_chain.invoke(test_input)
print(f"\nExplicit Result: {result2[:150]}...")

# Accessing chain components
print("\n4️⃣  Inspecting Chain Components:")
print("-" * 80)

print(f"\nChain steps: {len(chain.steps) if hasattr(chain, 'steps') else 'N/A'}")
print(f"First step: {chain.first if hasattr(chain, 'first') else 'N/A'}")
print(f"Last step: {chain.last if hasattr(chain, 'last') else 'N/A'}")

# Chaining multiple sequences
print("\n5️⃣  Chaining Sequences:")
print("-" * 80)

# First sequence
seq1 = prompt | llm

# Second sequence (extends first)
seq2 = seq1 | parser

# Third sequence (further extends)
from langchain_core.runnables import RunnableLambda

def add_prefix(text: str) -> str:
    return f"✨ {text}"

seq3 = seq2 | RunnableLambda(add_prefix)

result = seq3.invoke({"concept": "blockchain"})
print(f"\nFinal result: {result[:150]}...")

print("\n" + "=" * 80)
print("RUNNABLE SEQUENCE EXPLAINED")
print("=" * 80)
print("""
RunnableSequence:
- Created automatically when using |
- Connects multiple runnables in order
- Each output flows to next input

Structure:
    RunnableSequence(
        first=Component1,
        middle=[Component2, Component3],
        last=Component4
    )

Equivalent to:
    Component1 | Component2 | Component3 | Component4

Features:
✅ Automatic type handling
✅ Error propagation
✅ Streaming support
✅ Batch processing
✅ Async support

When output of one doesn't match input of next,
LangChain tries to adapt automatically!
""")
print("=" * 80)
