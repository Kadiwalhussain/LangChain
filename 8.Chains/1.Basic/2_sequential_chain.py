"""
Basic Chain #2: Sequential Chain
Chain multiple LLMChains together in sequence
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

print("=" * 80)
print("SEQUENTIAL CHAIN")
print("=" * 80)

# Initialize LLM
llm = OllamaLLM(model="mistral")

# First chain: Generate a topic
template1 = """Given a field of study, suggest an interesting topic.
Field: {field}

Topic:"""

prompt1 = PromptTemplate(input_variables=["field"], template=template1)
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Second chain: Write about the topic
template2 = """Given a topic, write a brief description (2-3 sentences).
Topic: {topic}

Description:"""

prompt2 = PromptTemplate(input_variables=["topic"], template=template2)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine into sequential chain
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True  # Shows intermediate steps
)

# Run the chain
input_text = "Machine Learning"

print(f"\n📝 Input Field: {input_text}")
print(f"\n🔄 Running Sequential Chain...\n")

result = overall_chain.invoke(input_text)

print(f"\n✅ Final Output:")
print(f"   {result['output']}")

print("\n" + "=" * 80)
print("SEQUENTIAL CHAIN FLOW")
print("=" * 80)
print("""
Input → Chain 1 → Output 1 → Chain 2 → Final Output

Example:
"Machine Learning" 
  → Chain 1 (suggest topic) 
  → "Neural Networks"
  → Chain 2 (describe topic)
  → "Neural Networks are computational models inspired by biological..."

Key Points:
✅ Output of chain N becomes input to chain N+1
✅ Useful for multi-step processing
✅ Each step builds on previous
✅ Verbose mode shows intermediate results
""")
print("=" * 80)
