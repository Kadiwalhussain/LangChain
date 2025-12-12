"""
Advanced Chain #3: Parallel Chain Execution
Execute multiple chains in parallel for efficiency
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import concurrent.futures
import time

print("=" * 80)
print("PARALLEL CHAIN EXECUTION")
print("=" * 80)

llm = OllamaLLM(model="mistral")

def create_analysis_chain(aspect: str) -> LLMChain:
    """Create a chain for analyzing a specific aspect"""
    template = f"""Analyze the {{aspect_name}} of this topic:
Topic: {{topic}}

{aspect} Analysis:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["topic", "aspect_name"]
    )
    return LLMChain(llm=llm, prompt=prompt)

# Create multiple analysis chains
chains = {
    "benefits": create_analysis_chain("Benefits and advantages"),
    "challenges": create_analysis_chain("Challenges and difficulties"),
    "future": create_analysis_chain("Future outlook and trends"),
    "examples": create_analysis_chain("Real-world examples and use cases")
}

def run_chain(name: str, chain: LLMChain, topic: str) -> tuple:
    """Run a single chain and return results"""
    start_time = time.time()
    print(f"  🔄 Starting {name} analysis...")
    
    result = chain.invoke({
        "topic": topic,
        "aspect_name": name
    })
    
    duration = time.time() - start_time
    print(f"  ✅ Completed {name} analysis ({duration:.2f}s)")
    
    return name, result['text'], duration

def parallel_analysis(topic: str):
    """Run all analysis chains in parallel"""
    print(f"\n📊 Analyzing topic: {topic}")
    print(f"\n⚡ Running {len(chains)} analyses in parallel...\n")
    
    start_time = time.time()
    results = {}
    
    # Execute chains in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chains)) as executor:
        futures = [
            executor.submit(run_chain, name, chain, topic)
            for name, chain in chains.items()
        ]
        
        for future in concurrent.futures.as_completed(futures):
            name, analysis, duration = future.result()
            results[name] = {
                "analysis": analysis,
                "duration": duration
            }
    
    total_time = time.time() - start_time
    
    return results, total_time

def sequential_analysis(topic: str):
    """Run all analysis chains sequentially (for comparison)"""
    print(f"\n📊 Analyzing topic: {topic}")
    print(f"\n🐌 Running {len(chains)} analyses sequentially...\n")
    
    start_time = time.time()
    results = {}
    
    for name, chain in chains.items():
        _, analysis, duration = run_chain(name, chain, topic)
        results[name] = {
            "analysis": analysis,
            "duration": duration
        }
    
    total_time = time.time() - start_time
    
    return results, total_time

# Test topic
topic = "Artificial Intelligence"

# Run parallel analysis
print("=" * 80)
print("PARALLEL EXECUTION")
print("=" * 80)
parallel_results, parallel_time = parallel_analysis(topic)

print("\n" + "=" * 80)
print("PARALLEL RESULTS")
print("=" * 80)

for name, data in parallel_results.items():
    print(f"\n📌 {name.upper()}:")
    print(f"   Time: {data['duration']:.2f}s")
    print(f"   Analysis: {data['analysis'][:150]}...")

print(f"\n⏱️  Total Parallel Time: {parallel_time:.2f}s")

# Compare with sequential (optional - comment out if too slow)
print("\n" + "=" * 80)
print("COMPARISON: PARALLEL vs SEQUENTIAL")
print("=" * 80)
print(f"""
Parallel Execution: {parallel_time:.2f}s
Sequential (estimated): {sum(d['duration'] for d in parallel_results.values()):.2f}s

Speedup: ~{sum(d['duration'] for d in parallel_results.values()) / parallel_time:.1f}x faster

Benefits of Parallel Execution:
✅ Faster processing
✅ Better resource utilization
✅ Handles multiple tasks simultaneously
✅ Scales with more chains
""")

print("\n" + "=" * 80)
print("PARALLEL CHAIN ARCHITECTURE")
print("=" * 80)
print("""
                    Input Topic
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    Benefits       Challenges         Future
     Chain           Chain            Chain
        ↓                ↓                ↓
    Analysis 1      Analysis 2      Analysis 3
        └────────────────┼────────────────┘
                         ↓
                 Combined Results

Use Cases:
- Multi-aspect analysis
- Parallel data processing
- Independent task execution
- Performance optimization

Note: True parallelism depends on LLM API support
""")
print("=" * 80)
