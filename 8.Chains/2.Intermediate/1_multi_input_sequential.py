"""
Intermediate Chain #1: Multiple Input Sequential Chain
Handle multiple inputs/outputs across sequential chains
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

print("=" * 80)
print("MULTIPLE INPUT SEQUENTIAL CHAIN")
print("=" * 80)

# Initialize LLM
llm = OllamaLLM(model="mistral")

# Chain 1: Analyze topic
template1 = """Analyze this topic and identify the key concept.
Topic: {topic}

Key Concept:"""

prompt1 = PromptTemplate(input_variables=["topic"], template=template1)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="key_concept")

# Chain 2: Generate questions using both original topic and key concept
template2 = """Based on the topic and key concept, generate 3 questions.
Topic: {topic}
Key Concept: {key_concept}

Questions:"""

prompt2 = PromptTemplate(
    input_variables=["topic", "key_concept"],
    template=template2
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="questions")

# Chain 3: Create summary using all previous outputs
template3 = """Create a brief summary.
Topic: {topic}
Key Concept: {key_concept}
Questions: {questions}

Summary:"""

prompt3 = PromptTemplate(
    input_variables=["topic", "key_concept", "questions"],
    template=template3
)
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="summary")

# Combine into sequential chain
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["topic"],
    output_variables=["key_concept", "questions", "summary"],
    verbose=True
)

# Run the chain
result = overall_chain.invoke({"topic": "Machine Learning"})

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\n📌 Key Concept:\n{result['key_concept']}")
print(f"\n❓ Questions:\n{result['questions']}")
print(f"\n📝 Summary:\n{result['summary']}")

print("\n" + "=" * 80)
print("FLOW DIAGRAM")
print("=" * 80)
print("""
Input: {topic}
    ↓
Chain 1: topic → key_concept
    ↓
Chain 2: topic + key_concept → questions
    ↓
Chain 3: topic + key_concept + questions → summary
    ↓
Output: {key_concept, questions, summary}

Key Feature: Later chains can access outputs from ALL previous chains
""")
print("=" * 80)
