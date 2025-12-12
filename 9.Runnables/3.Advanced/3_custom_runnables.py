"""
Runnable Advanced #3: Custom Runnables
Create production-ready custom Runnable classes
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Iterator, Optional, Dict, List
import time

print("=" * 80)
print("CUSTOM RUNNABLES - BUILD YOUR OWN")
print("=" * 80)

# Example 1: Simple Custom Runnable
print("\n1️⃣  Simple Custom Runnable:")
print("-" * 80)

class UpperCaseRunnable(Runnable[str, str]):
    """Convert input to uppercase"""
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> str:
        """Synchronous invocation"""
        print(f"  🔄 Converting to uppercase: {input[:30]}...")
        return input.upper()
    
    def stream(self, input: str, config: Optional[RunnableConfig] = None) -> Iterator[str]:
        """Streaming support (char by char)"""
        for char in input.upper():
            yield char
    
    def batch(self, inputs: List[str], config: Optional[RunnableConfig] = None) -> List[str]:
        """Batch processing"""
        return [self.invoke(inp, config) for inp in inputs]

uppercase = UpperCaseRunnable()

# Test invoke
result = uppercase.invoke("hello world")
print(f"Result: {result}")

# Test in chain
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()
prompt = PromptTemplate.from_template("Say: {text}")

chain = prompt | llm | parser | uppercase
result = chain.invoke({"text": "artificial intelligence"})
print(f"Chain result: {result[:100]}...")

# Example 2: Validation Runnable
print("\n\n2️⃣  Validation Runnable:")
print("-" * 80)

class ValidatedRunnable(Runnable[str, Dict[str, Any]]):
    """Validate and enhance LLM output"""
    
    def __init__(self, min_length: int = 50, max_length: int = 500):
        self.min_length = min_length
        self.max_length = max_length
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Validate and return enhanced output"""
        word_count = len(input.split())
        char_count = len(input)
        
        # Validation
        is_valid = self.min_length <= char_count <= self.max_length
        
        issues = []
        if char_count < self.min_length:
            issues.append(f"Too short ({char_count} < {self.min_length})")
        if char_count > self.max_length:
            issues.append(f"Too long ({char_count} > {self.max_length})")
        
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {status}: {word_count} words, {char_count} chars")
        
        return {
            "text": input,
            "is_valid": is_valid,
            "word_count": word_count,
            "char_count": char_count,
            "issues": issues,
            "metadata": {
                "validated_at": time.time(),
                "min_length": self.min_length,
                "max_length": self.max_length
            }
        }

validator = ValidatedRunnable(min_length=30, max_length=200)

# Test validation
validation_chain = (
    PromptTemplate.from_template("Explain {topic} briefly:")
    | llm
    | parser
    | validator
)

result = validation_chain.invoke({"topic": "Python"})
print(f"\nValidation result:")
print(f"  Valid: {result['is_valid']}")
print(f"  Words: {result['word_count']}")
print(f"  Issues: {result['issues']}")
print(f"  Text: {result['text'][:80]}...")

# Example 3: Caching Runnable
print("\n\n3️⃣  Caching Runnable:")
print("-" * 80)

class CachedRunnable(Runnable[str, str]):
    """Cache LLM responses for repeated queries"""
    
    def __init__(self, llm_chain: Runnable):
        self.llm_chain = llm_chain
        self.cache: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> str:
        """Check cache before invoking LLM"""
        cache_key = input.lower().strip()
        
        if cache_key in self.cache:
            self.hits += 1
            print(f"  💾 Cache HIT ({self.hits} hits, {self.misses} misses)")
            return self.cache[cache_key]
        
        self.misses += 1
        print(f"  🔄 Cache MISS ({self.hits} hits, {self.misses} misses)")
        
        # Invoke LLM
        result = self.llm_chain.invoke(input, config)
        
        # Store in cache
        self.cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        print(f"  🗑️  Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }

# Create cached chain
base_chain = PromptTemplate.from_template("Explain: {topic}") | llm | parser
cached = CachedRunnable(base_chain)

# Test caching
topics = ["Python", "JavaScript", "Python", "Python", "Go", "JavaScript"]

for topic in topics:
    print(f"\nQuery: {topic}")
    result = cached.invoke(topic)
    print(f"Result: {result[:60]}...")

print(f"\n📊 Cache Statistics:")
stats = cached.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

# Example 4: Retry Runnable
print("\n\n4️⃣  Retry Runnable:")
print("-" * 80)

class RetryRunnable(Runnable[Any, Any]):
    """Automatically retry failed operations"""
    
    def __init__(self, runnable: Runnable, max_retries: int = 3, delay: float = 0.5):
        self.runnable = runnable
        self.max_retries = max_retries
        self.delay = delay
    
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Invoke with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"  🔄 Attempt {attempt + 1}/{self.max_retries}")
                result = self.runnable.invoke(input, config)
                
                if attempt > 0:
                    print(f"  ✅ Succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_error = e
                print(f"  ❌ Attempt {attempt + 1} failed: {str(e)[:50]}")
                
                if attempt < self.max_retries - 1:
                    print(f"  ⏳ Waiting {self.delay}s before retry...")
                    time.sleep(self.delay)
        
        print(f"  💥 All {self.max_retries} attempts failed")
        raise last_error

# Simulate unreliable operation
class UnreliableRunnable(Runnable[str, str]):
    """Runnable that fails sometimes"""
    
    def __init__(self):
        self.attempt_count = 0
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> str:
        self.attempt_count += 1
        
        # Fail first 2 attempts, succeed on 3rd
        if self.attempt_count < 3:
            raise ConnectionError(f"Simulated failure #{self.attempt_count}")
        
        return f"Success! Input was: {input}"

unreliable = UnreliableRunnable()
retry_chain = RetryRunnable(unreliable, max_retries=5, delay=0.3)

print("\nTesting retry logic:")
result = retry_chain.invoke("test data")
print(f"\n📝 Final result: {result}")

# Example 5: Monitoring Runnable
print("\n\n5️⃣  Monitoring Runnable:")
print("-" * 80)

class MonitoredRunnable(Runnable[Any, Any]):
    """Monitor and log chain execution"""
    
    def __init__(self, runnable: Runnable, name: str = "chain"):
        self.runnable = runnable
        self.name = name
        self.executions: List[Dict[str, Any]] = []
    
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Invoke with monitoring"""
        start_time = time.time()
        
        print(f"  ▶️  Starting {self.name}")
        
        try:
            result = self.runnable.invoke(input, config)
            duration = time.time() - start_time
            
            # Log execution
            self.executions.append({
                "input": str(input)[:50],
                "success": True,
                "duration": duration,
                "timestamp": time.time()
            })
            
            print(f"  ✅ Completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log failure
            self.executions.append({
                "input": str(input)[:50],
                "success": False,
                "error": str(e)[:50],
                "duration": duration,
                "timestamp": time.time()
            })
            
            print(f"  ❌ Failed after {duration:.2f}s")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        if not self.executions:
            return {"message": "No executions yet"}
        
        successful = [e for e in self.executions if e["success"]]
        failed = [e for e in self.executions if not e["success"]]
        
        avg_duration = sum(e["duration"] for e in self.executions) / len(self.executions)
        
        return {
            "total_executions": len(self.executions),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": f"{len(successful)/len(self.executions)*100:.1f}%",
            "avg_duration": f"{avg_duration:.2f}s"
        }

# Create monitored chain
base = PromptTemplate.from_template("Explain: {topic}") | llm | parser
monitored = MonitoredRunnable(base, name="ExplainChain")

# Execute multiple times
for topic in ["AI", "Blockchain", "Python"]:
    print(f"\n📝 Topic: {topic}")
    result = monitored.invoke({"topic": topic})
    print(f"   Result: {result[:60]}...")

print(f"\n📊 Execution Metrics:")
metrics = monitored.get_metrics()
for key, value in metrics.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 80)
print("CUSTOM RUNNABLES EXPLAINED")
print("=" * 80)
print("""
Custom Runnables extend LangChain functionality:

Base Class:
    from langchain_core.runnables import Runnable
    
    class MyRunnable(Runnable[InputType, OutputType]):
        def invoke(self, input, config=None):
            # Your logic
            return output

Required Methods:

1. invoke(input, config) → output
   - Synchronous execution
   - Main method

2. Optional Methods:
   - stream(input, config) → Iterator[output]
   - batch(inputs, config) → List[output]
   - ainvoke(input, config) → Awaitable[output] (async)
   - astream(input, config) → AsyncIterator[output]

Use Cases:

1. Validation:
   - Input/output validation
   - Schema enforcement
   - Type checking

2. Caching:
   - Response caching
   - Deduplication
   - Performance optimization

3. Retry Logic:
   - Automatic retries
   - Exponential backoff
   - Error recovery

4. Monitoring:
   - Execution tracking
   - Performance metrics
   - Logging

5. Rate Limiting:
   - API quota management
   - Throttling
   - Queue management

6. Transformation:
   - Data preprocessing
   - Format conversion
   - Enrichment

7. Routing:
   - Dynamic chain selection
   - Load balancing
   - Failover

Benefits:
✅ Reusable components
✅ Composable with any chain
✅ Full control over behavior
✅ Type-safe
✅ Testable

Best Practices:
- Keep single responsibility
- Handle errors gracefully
- Add logging/monitoring
- Support config parameter
- Implement streaming when possible
- Write unit tests
- Document behavior
- Version your runnables

Production Patterns:

1. Wrapper Runnable:
   class Wrapper(Runnable):
       def __init__(self, inner: Runnable):
           self.inner = inner
       def invoke(self, input, config):
           # Pre-process
           result = self.inner.invoke(input, config)
           # Post-process
           return result

2. Composite Runnable:
   Combines multiple runnables internally

3. Stateful Runnable:
   Maintains state across invocations

4. Async Runnable:
   Implements ainvoke for async operations

Advanced Features:
- Config propagation
- Error handling
- Batch processing
- Streaming support
- Context management
- Resource cleanup
""")
print("=" * 80)
