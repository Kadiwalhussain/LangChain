"""
Advanced Chain #2: Conditional Sequential Chain
Chain that adapts based on intermediate results
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

print("=" * 80)
print("CONDITIONAL SEQUENTIAL CHAIN")
print("=" * 80)

llm = OllamaLLM(model="mistral")

def conditional_sequential_chain(user_query: str):
    """
    Chain that changes behavior based on intermediate results
    """
    
    # Step 1: Classify the query
    print(f"\n1️⃣  Classifying query: {user_query}")
    
    classify_template = """Classify this query into ONE category: factual, creative, or analytical
    
Query: {query}

Category (respond with ONLY one word):"""
    
    classify_prompt = PromptTemplate(template=classify_template, input_variables=["query"])
    classify_chain = LLMChain(llm=llm, prompt=classify_prompt)
    
    classification = classify_chain.invoke({"query": user_query})
    category = classification['text'].strip().lower()
    print(f"   Category: {category}")
    
    # Step 2: Choose appropriate chain based on classification
    print(f"\n2️⃣  Selecting appropriate processing chain...")
    
    if "factual" in category:
        print(f"   Using: Factual Answer Chain")
        template = """Provide a factual, accurate answer with sources if possible:
{query}

Factual Answer:"""
        
    elif "creative" in category:
        print(f"   Using: Creative Writing Chain")
        template = """Respond creatively and imaginatively to this:
{query}

Creative Response:"""
        
    else:  # analytical
        print(f"   Using: Analytical Chain")
        template = """Provide a detailed analytical response with pros and cons:
{query}

Analysis:"""
    
    # Step 3: Execute chosen chain
    print(f"\n3️⃣  Generating response...")
    
    response_prompt = PromptTemplate(template=template, input_variables=["query"])
    response_chain = LLMChain(llm=llm, prompt=response_prompt)
    
    response = response_chain.invoke({"query": user_query})
    
    # Step 4: Post-process based on length
    print(f"\n4️⃣  Post-processing...")
    
    answer = response['text']
    word_count = len(answer.split())
    
    if word_count < 30:
        print(f"   Response too short ({word_count} words), expanding...")
        
        expand_template = """Expand this answer with more details:
Original: {answer}

Expanded Answer:"""
        
        expand_prompt = PromptTemplate(template=expand_template, input_variables=["answer"])
        expand_chain = LLMChain(llm=llm, prompt=expand_prompt)
        
        final_response = expand_chain.invoke({"answer": answer})
        answer = final_response['text']
    else:
        print(f"   Response length adequate ({word_count} words)")
    
    return {
        "category": category,
        "answer": answer,
        "word_count": len(answer.split())
    }

# Test queries
queries = [
    "What is the population of Tokyo?",
    "Write a poem about coding",
    "Should companies adopt remote work?"
]

print("\n🧪 Testing Conditional Chain:\n")

for query in queries:
    print("\n" + "=" * 80)
    result = conditional_sequential_chain(query)
    print(f"\n📊 RESULTS:")
    print(f"   Category: {result['category']}")
    print(f"   Word Count: {result['word_count']}")
    print(f"\n💬 Answer:\n{result['answer'][:200]}...")

print("\n" + "=" * 80)
print("CONDITIONAL CHAIN FLOW")
print("=" * 80)
print("""
Input Query
    ↓
Classify Query
    ↓
    ├─ Factual? → Factual Chain
    ├─ Creative? → Creative Chain
    └─ Analytical? → Analytical Chain
    ↓
Generate Response
    ↓
Check Length
    ├─ Too Short? → Expand Chain
    └─ Adequate? → Return
    ↓
Final Answer

Key Features:
✅ Dynamic chain selection
✅ Adaptive behavior
✅ Post-processing validation
✅ Self-correcting
""")
print("=" * 80)
