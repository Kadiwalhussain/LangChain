"""
Runnable Advanced #1: Dynamic Routing with RunnableLambda
Implement intelligent routing based on content analysis
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from typing import Dict, Any

print("=" * 80)
print("DYNAMIC ROUTING - INTELLIGENT CHAIN SELECTION")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Example 1: LLM-based routing
print("\n1️⃣  LLM-Based Dynamic Routing:")
print("-" * 80)

def classify_query(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to classify query type"""
    query = input_dict["query"]
    
    classifier_prompt = f"""Classify this query into ONE category:
- technical: Programming, code, bugs, errors
- creative: Writing, brainstorming, ideas
- factual: Questions about facts, definitions, history
- analytical: Data analysis, comparisons, evaluations

Query: {query}

Respond with ONLY the category name (one word):"""
    
    category = llm.invoke(classifier_prompt).strip().lower()
    
    # Ensure valid category
    valid_categories = ["technical", "creative", "factual", "analytical"]
    if category not in valid_categories:
        category = "factual"  # default
    
    print(f"  🏷️  Classified as: {category}")
    
    return {
        "query": query,
        "category": category
    }

def route_to_chain(classified_dict: Dict[str, Any]) -> str:
    """Route to appropriate specialized chain"""
    query = classified_dict["query"]
    category = classified_dict["category"]
    
    # Define specialized prompts
    prompts = {
        "technical": """You are a senior software engineer. 
Technical Query: {query}
Provide detailed technical solution with code examples:""",
        
        "creative": """You are a creative writer and brainstorming expert.
Creative Request: {query}
Provide imaginative and innovative response:""",
        
        "factual": """You are a knowledgeable encyclopedic assistant.
Factual Question: {query}
Provide accurate, well-researched answer:""",
        
        "analytical": """You are a data analyst and critical thinker.
Analytical Query: {query}
Provide structured analysis with pros/cons:"""
    }
    
    selected_prompt = prompts.get(category, prompts["factual"])
    prompt_template = PromptTemplate.from_template(selected_prompt)
    
    # Execute specialized chain
    chain = prompt_template | llm | parser
    result = chain.invoke({"query": query})
    
    return f"[{category.upper()}] {result}"

# Create dynamic routing chain
dynamic_router = (
    RunnableLambda(classify_query)
    | RunnableLambda(route_to_chain)
)

# Test with different query types
test_queries = [
    "How do I fix a Python syntax error?",
    "Write a poem about AI",
    "What is the capital of France?",
    "Compare Python and JavaScript for web development"
]

for query in test_queries:
    print(f"\n❓ Query: {query}")
    result = dynamic_router.invoke({"query": query})
    print(f"📝 Response: {result[:150]}...\n")
    print("-" * 80)

# Example 2: Multi-factor routing
print("\n\n2️⃣  Multi-Factor Routing (Length + Complexity):")
print("-" * 80)

def analyze_query_complexity(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze multiple factors to determine routing"""
    query = input_dict["query"]
    
    # Factor 1: Length
    word_count = len(query.split())
    is_long = word_count > 15
    
    # Factor 2: Complexity (has technical terms)
    technical_terms = ["algorithm", "database", "api", "function", "class", "error"]
    is_complex = any(term in query.lower() for term in technical_terms)
    
    # Factor 3: Question type
    is_howto = query.lower().startswith(("how", "what", "why"))
    
    # Determine route
    if is_complex and is_long:
        route = "expert"
    elif is_howto:
        route = "tutorial"
    elif word_count < 5:
        route = "quick"
    else:
        route = "standard"
    
    print(f"  📊 Analysis: words={word_count}, complex={is_complex}, howto={is_howto}")
    print(f"  🎯 Route: {route}")
    
    return {
        "query": query,
        "route": route,
        "word_count": word_count,
        "is_complex": is_complex
    }

def execute_routed_chain(analyzed_dict: Dict[str, Any]) -> str:
    """Execute appropriate chain based on analysis"""
    query = analyzed_dict["query"]
    route = analyzed_dict["route"]
    
    chains = {
        "expert": PromptTemplate.from_template(
            "Provide expert-level detailed analysis:\n{query}"
        ) | llm | parser,
        
        "tutorial": PromptTemplate.from_template(
            "Provide step-by-step tutorial:\n{query}"
        ) | llm | parser,
        
        "quick": PromptTemplate.from_template(
            "Provide brief answer:\n{query}"
        ) | llm | parser,
        
        "standard": PromptTemplate.from_template(
            "Provide standard answer:\n{query}"
        ) | llm | parser
    }
    
    selected_chain = chains[route]
    result = selected_chain.invoke({"query": query})
    
    return f"[{route.upper()}] {result}"

complexity_router = (
    RunnableLambda(analyze_query_complexity)
    | RunnableLambda(execute_routed_chain)
)

test_cases = [
    "What is Python?",
    "How do I implement a binary search tree algorithm in Python with error handling?",
    "Explain machine learning"
]

for query in test_cases:
    print(f"\n❓ {query}")
    result = complexity_router.invoke({"query": query})
    print(f"📝 {result[:120]}...\n")

# Example 3: Confidence-based routing
print("\n\n3️⃣  Confidence-Based Routing:")
print("-" * 80)

def get_confidence_and_answer(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Get initial answer and confidence level"""
    query = input_dict["query"]
    
    # Get initial answer
    initial_prompt = PromptTemplate.from_template(
        "Answer this question: {query}\n\nYour answer:"
    )
    initial_chain = initial_prompt | llm | parser
    answer = initial_chain.invoke({"query": query})
    
    # Estimate confidence (simple heuristic)
    confidence_indicators = {
        "high": ["definitely", "certainly", "clearly", "obviously"],
        "medium": ["probably", "likely", "generally", "typically"],
        "low": ["maybe", "possibly", "might", "uncertain", "unclear"]
    }
    
    answer_lower = answer.lower()
    if any(word in answer_lower for word in confidence_indicators["high"]):
        confidence = "high"
    elif any(word in answer_lower for word in confidence_indicators["low"]):
        confidence = "low"
    else:
        confidence = "medium"
    
    print(f"  📈 Confidence: {confidence}")
    
    return {
        "query": query,
        "answer": answer,
        "confidence": confidence
    }

def enhance_if_needed(result_dict: Dict[str, Any]) -> str:
    """Enhance answer if confidence is low"""
    confidence = result_dict["confidence"]
    answer = result_dict["answer"]
    query = result_dict["query"]
    
    if confidence == "low":
        print(f"  🔄 Enhancing low-confidence answer...")
        enhance_prompt = PromptTemplate.from_template(
            "The following answer seems uncertain:\n{answer}\n\n"
            "Provide a more confident and detailed answer to: {query}"
        )
        enhance_chain = enhance_prompt | llm | parser
        enhanced = enhance_chain.invoke({"query": query, "answer": answer})
        return f"[ENHANCED] {enhanced}"
    
    return f"[CONFIDENT] {answer}"

confidence_router = (
    RunnableLambda(get_confidence_and_answer)
    | RunnableLambda(enhance_if_needed)
)

result = confidence_router.invoke({"query": "What might be the future of AI?"})
print(f"\n{result[:200]}...\n")

# Example 4: Parallel analysis then route
print("\n\n4️⃣  Parallel Analysis + Dynamic Routing:")
print("-" * 80)

def parallel_analysis(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze multiple aspects in parallel"""
    query = input_dict["query"]
    
    # Create parallel analyzers
    analyzers = RunnableParallel(
        sentiment=RunnableLambda(lambda x: "positive" if "good" in x["query"].lower() 
                                 or "best" in x["query"].lower() else "neutral"),
        length=RunnableLambda(lambda x: "long" if len(x["query"].split()) > 10 else "short"),
        has_code=RunnableLambda(lambda x: "yes" if "code" in x["query"].lower() 
                                or "python" in x["query"].lower() else "no")
    )
    
    analysis = analyzers.invoke({"query": query})
    
    print(f"  🔍 Sentiment: {analysis['sentiment']}")
    print(f"  🔍 Length: {analysis['length']}")
    print(f"  🔍 Has code: {analysis['has_code']}")
    
    return {
        "query": query,
        "analysis": analysis
    }

def route_based_on_analysis(analyzed_dict: Dict[str, Any]) -> str:
    """Route based on parallel analysis results"""
    query = analyzed_dict["query"]
    analysis = analyzed_dict["analysis"]
    
    # Decision logic
    if analysis["has_code"] == "yes":
        route = "code_expert"
    elif analysis["sentiment"] == "positive" and analysis["length"] == "long":
        route = "detailed_positive"
    else:
        route = "standard"
    
    print(f"  ➡️  Routed to: {route}")
    
    prompts = {
        "code_expert": "As a coding expert, answer: {query}",
        "detailed_positive": "Provide enthusiastic detailed response: {query}",
        "standard": "Answer: {query}"
    }
    
    chain = PromptTemplate.from_template(prompts[route]) | llm | parser
    return chain.invoke({"query": query})

parallel_router = (
    RunnableLambda(parallel_analysis)
    | RunnableLambda(route_based_on_analysis)
)

test_query = "What's the best Python code for sorting?"
print(f"\n❓ {test_query}")
result = parallel_router.invoke({"query": test_query})
print(f"📝 {result[:150]}...\n")

print("\n" + "=" * 80)
print("DYNAMIC ROUTING EXPLAINED")
print("=" * 80)
print("""
Dynamic Routing selects chains based on runtime analysis:

Architecture:
    Input → Analyzer → Router → Selected Chain → Output

Types of Routing:

1. Content-Based:
   - LLM classification
   - Keyword detection
   - Pattern matching

2. Multi-Factor:
   - Query length
   - Complexity
   - User preferences
   - Historical data

3. Confidence-Based:
   - Initial answer quality
   - Uncertainty detection
   - Auto-enhancement

4. Parallel Analysis:
   - Multiple simultaneous checks
   - Combined decision logic
   - Complex routing rules

Implementation Patterns:

A. Simple Router:
    analyze | route | execute

B. Feedback Router:
    execute | validate | re-route if needed

C. Multi-Stage Router:
    classify | analyze | route | execute | validate

D. Parallel Router:
    parallel_analyze | combine | route | execute

Benefits:
✅ Intelligent chain selection
✅ Optimal resource usage
✅ Better accuracy
✅ Cost optimization
✅ Adaptive behavior

Use Cases:
- Customer support triage
- Content moderation
- Multi-domain QA
- Language routing
- Difficulty adaptation
- Cost-aware processing

Best Practices:
- Fast classification (< 100ms)
- Clear routing logic
- Fallback for unknown cases
- Log routing decisions
- Monitor distribution
- A/B test routing rules
""")
print("=" * 80)
