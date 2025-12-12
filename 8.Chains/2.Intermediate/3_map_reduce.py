"""
Intermediate Chain #3: Simplified Map-Reduce Pattern
Process multiple documents in parallel and combine results
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

print("=" * 80)
print("MAP-REDUCE PATTERN (Simplified)")
print("=" * 80)

# Initialize LLM
llm = OllamaLLM(model="mistral")

# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "JavaScript is widely used for web development and runs in browsers.",
    "Java is a popular language for enterprise applications and Android.",
]

# MAP PHASE: Summarize each document
print("\n📊 MAP PHASE - Processing each document:")
print("=" * 80)

map_template = """Summarize this text in one short sentence:
{text}

Summary:"""

map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Process each document
summaries = []
for i, doc in enumerate(documents, 1):
    print(f"\nDocument {i}: {doc}")
    result = map_chain.invoke({"text": doc})
    summary = result['text'].strip()
    summaries.append(summary)
    print(f"Summary {i}: {summary}")

# REDUCE PHASE: Combine all summaries
print("\n\n🔄 REDUCE PHASE - Combining summaries:")
print("=" * 80)

reduce_template = """Combine these summaries into one coherent paragraph:

{summaries}

Combined Summary:"""

reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["summaries"])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Combine summaries
combined_summaries = "\n".join([f"{i}. {s}" for i, s in enumerate(summaries, 1)])
final_result = reduce_chain.invoke({"summaries": combined_summaries})

print(f"\n✅ Final Combined Summary:")
print(final_result['text'])

print("\n" + "=" * 80)
print("MAP-REDUCE FLOW")
print("=" * 80)
print("""
Documents: [Doc1, Doc2, Doc3]
        ↓
    MAP PHASE (Parallel processing)
        ↓
[Summary1, Summary2, Summary3]
        ↓
    REDUCE PHASE (Combine)
        ↓
    Final Summary

Use Cases:
✅ Summarize multiple articles
✅ Process large documents (split into chunks)
✅ Analyze multiple reviews
✅ Aggregate data from multiple sources

Benefits:
✅ Parallel processing (faster)
✅ Handles large inputs
✅ Structured approach
✅ Scalable
""")
print("=" * 80)
