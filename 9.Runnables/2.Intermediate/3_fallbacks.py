"""
Runnable Intermediate #3: Fallbacks and Error Handling
Implement retry logic and backup chains
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

print("=" * 80)
print("FALLBACKS AND ERROR HANDLING")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Example 1: Basic fallback with_fallbacks()
print("\n1️⃣  Basic Fallback:")
print("-" * 80)

# Primary chain that might fail
def unreliable_chain(input_dict: dict) -> str:
    """Chain that sometimes fails"""
    import random
    if random.random() < 0.5:  # 50% chance of failure
        raise ValueError("Primary chain failed!")
    return f"✅ Primary chain succeeded for: {input_dict['query']}"

# Backup chain
def backup_chain(input_dict: dict) -> str:
    """Reliable backup chain"""
    return f"🔄 Backup chain activated for: {input_dict['query']}"

primary = RunnableLambda(unreliable_chain)
backup = RunnableLambda(backup_chain)

# Create chain with fallback
chain_with_fallback = primary.with_fallbacks([backup])

# Test multiple times
print("Testing chain with fallback (5 attempts):\n")
for i in range(5):
    try:
        result = chain_with_fallback.invoke({"query": f"Test {i+1}"})
        print(f"Attempt {i+1}: {result}")
    except Exception as e:
        print(f"Attempt {i+1}: ❌ All chains failed - {e}")

# Example 2: Multiple fallbacks
print("\n\n2️⃣  Multiple Fallback Chains:")
print("-" * 80)

def primary_llm_chain(input_dict: dict) -> str:
    """Primary LLM chain"""
    prompt = PromptTemplate.from_template("Explain {topic} in detail:")
    chain = prompt | llm | parser
    result = chain.invoke(input_dict)
    return f"🥇 Primary: {result[:100]}..."

def secondary_llm_chain(input_dict: dict) -> str:
    """Secondary LLM chain with simpler prompt"""
    prompt = PromptTemplate.from_template("What is {topic}?")
    chain = prompt | llm | parser
    result = chain.invoke(input_dict)
    return f"🥈 Secondary: {result[:100]}..."

def cache_chain(input_dict: dict) -> str:
    """Cached responses fallback"""
    cache = {
        "python": "Python is a high-level programming language.",
        "javascript": "JavaScript is a programming language for web development."
    }
    topic = input_dict["topic"].lower()
    if topic in cache:
        return f"💾 Cache: {cache[topic]}"
    return f"📝 Cache: No cached response for {topic}"

# Create multi-level fallback
primary_chain = RunnableLambda(primary_llm_chain)
secondary_chain = RunnableLambda(secondary_llm_chain)
cache_fallback = RunnableLambda(cache_chain)

multi_fallback_chain = primary_chain.with_fallbacks([secondary_chain, cache_fallback])

result = multi_fallback_chain.invoke({"topic": "Python"})
print(result)

# Example 3: Fallback with validation
print("\n\n3️⃣  Fallback with Output Validation:")
print("-" * 80)

def validate_length(text: str, min_length: int = 50) -> bool:
    """Validate output length"""
    return len(text) >= min_length

def primary_chain_validated(input_dict: dict) -> str:
    """Primary chain with potential short output"""
    prompt = PromptTemplate.from_template("Brief answer about {topic}:")
    chain = prompt | llm | parser
    result = chain.invoke(input_dict)
    
    if not validate_length(result, min_length=50):
        raise ValueError(f"Output too short: {len(result)} chars")
    
    return f"✅ Validated primary: {result}"

def detailed_backup_chain(input_dict: dict) -> str:
    """Backup that guarantees longer output"""
    prompt = PromptTemplate.from_template(
        "Provide a comprehensive explanation of {topic} with examples and details:"
    )
    chain = prompt | llm | parser
    result = chain.invoke(input_dict)
    return f"📚 Detailed backup: {result}"

validated_chain = (
    RunnableLambda(primary_chain_validated)
    .with_fallbacks([RunnableLambda(detailed_backup_chain)])
)

result = validated_chain.invoke({"topic": "AI"})
print(result[:200] + "...")

# Example 4: Conditional fallback routing
print("\n\n4️⃣  Conditional Fallback:")
print("-" * 80)

class ChainFailureError(Exception):
    """Custom error for chain failures"""
    pass

def fast_chain(input_dict: dict) -> str:
    """Fast but might fail"""
    import random
    if random.random() < 0.3:  # 30% failure
        raise ChainFailureError("Fast chain overloaded")
    return f"⚡ Fast: Quick response for {input_dict['query']}"

def accurate_chain(input_dict: dict) -> str:
    """Slower but accurate"""
    import time
    time.sleep(0.5)  # Simulate slower processing
    return f"🎯 Accurate: Detailed response for {input_dict['query']}"

def simple_chain(input_dict: dict) -> str:
    """Simple fallback"""
    return f"💬 Simple: Basic response for {input_dict['query']}"

# Chain with multiple fallbacks
robust_chain = (
    RunnableLambda(fast_chain)
    .with_fallbacks([
        RunnableLambda(accurate_chain),
        RunnableLambda(simple_chain)
    ])
)

print("Testing robust chain (10 attempts):\n")
for i in range(10):
    result = robust_chain.invoke({"query": f"Request {i+1}"})
    print(f"{i+1:2d}. {result}")

# Example 5: Retry with exponential backoff
print("\n\n5️⃣  Retry with Exponential Backoff:")
print("-" * 80)

def chain_with_retry(input_dict: dict) -> str:
    """Chain with built-in retry logic"""
    import time
    import random
    
    max_retries = 3
    base_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Simulate operation that might fail
            if random.random() < 0.6:  # 60% failure rate
                raise ConnectionError(f"Attempt {attempt + 1} failed")
            
            return f"✅ Success on attempt {attempt + 1}"
            
        except ConnectionError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"  ⚠️  {e} - Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"  ❌ All {max_retries} attempts failed")
                raise

retry_chain = RunnableLambda(chain_with_retry)

# Test retry mechanism
for i in range(3):
    print(f"\nTest {i+1}:")
    try:
        result = retry_chain.invoke({"query": "test"})
        print(f"  {result}")
    except ConnectionError:
        print(f"  ❌ Chain failed after all retries")

print("\n" + "=" * 80)
print("FALLBACKS AND ERROR HANDLING EXPLAINED")
print("=" * 80)
print("""
Fallbacks provide resilience and reliability:

Basic Syntax:
    primary_chain.with_fallbacks([backup1, backup2, backup3])

Execution Flow:
    Input → Primary ─Success→ Output
              │
             Fails
              ↓
            Backup1 ─Success→ Output
              │
             Fails
              ↓
            Backup2 ─Success→ Output
              │
             Fails
              ↓
            Exception raised

Use Cases:
✅ Multiple LLM providers
✅ Degraded functionality
✅ Cached responses
✅ Rate limit handling
✅ Network failures
✅ Model unavailability

Patterns:

1. Performance Fallback:
   fast_model → accurate_model → simple_rules

2. Cost Fallback:
   cheap_api → expensive_api → local_model

3. Quality Fallback:
   complex_chain → simple_chain → hardcoded

4. Availability Fallback:
   primary_service → backup_service → cache

Best Practices:
- Order by preference (best first)
- Include reliable final fallback
- Log which fallback was used
- Monitor fallback frequency
- Combine with retries for transient failures

Error Types to Handle:
- Network errors (retry)
- Rate limits (fallback + backoff)
- Invalid outputs (validation + fallback)
- Service unavailable (switch provider)
- Timeouts (faster fallback)

Fallbacks vs Retries:
- Retries: Try same operation again
- Fallbacks: Try different operation
- Often used together!
""")
print("=" * 80)
