"""
Runnable Intermediate #1: RunnableParallel
Execute multiple chains simultaneously
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
import time

print("=" * 80)
print("RUNNABLE PARALLEL - CONCURRENT EXECUTION")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Example 1: Basic Parallel Execution
print("\n1️⃣  Basic RunnableParallel:")
print("-" * 80)

# Define multiple analysis chains
summary_prompt = PromptTemplate.from_template("Summarize {topic} in one sentence:")
benefits_prompt = PromptTemplate.from_template("List 3 benefits of {topic}:")
challenges_prompt = PromptTemplate.from_template("List 3 challenges of {topic}:")

summary_chain = summary_prompt | llm | parser
benefits_chain = benefits_prompt | llm | parser
challenges_chain = challenges_prompt | llm | parser

# Create parallel runnable
parallel = RunnableParallel(
    summary=summary_chain,
    benefits=benefits_chain,
    challenges=challenges_chain
)

start_time = time.time()
result = parallel.invoke({"topic": "Artificial Intelligence"})
duration = time.time() - start_time

print(f"\n⏱️  Execution time: {duration:.2f} seconds\n")
print(f"Summary:\n{result['summary']}\n")
print(f"Benefits:\n{result['benefits']}\n")
print(f"Challenges:\n{result['challenges']}\n")

# Example 2: Using dict syntax (shorthand)
print("\n2️⃣  Dict Syntax (Shorthand):")
print("-" * 80)

# More concise way to create parallel
parallel_dict = {
    "summary": summary_chain,
    "benefits": benefits_chain,
    "challenges": challenges_chain
}

parallel2 = RunnableParallel(parallel_dict)
# Or even shorter:
# parallel2 = RunnableParallel(**parallel_dict)

result2 = parallel2.invoke({"topic": "Blockchain"})
print(f"\nSummary: {result2['summary'][:100]}...")
print(f"Benefits: {result2['benefits'][:100]}...")

# Example 3: Mixed inputs
print("\n3️⃣  Parallel with Different Inputs:")
print("-" * 80)

def get_length(input_dict: dict) -> int:
    """Get length of topic name"""
    return len(input_dict["topic"])

def get_uppercase(input_dict: dict) -> str:
    """Convert topic to uppercase"""
    return input_dict["topic"].upper()

parallel3 = RunnableParallel(
    original_topic=lambda x: x["topic"],
    length=RunnableLambda(get_length),
    uppercase=RunnableLambda(get_uppercase),
    analysis=summary_chain
)

result3 = parallel3.invoke({"topic": "Python"})
print(f"\nOriginal: {result3['original_topic']}")
print(f"Length: {result3['length']}")
print(f"Uppercase: {result3['uppercase']}")
print(f"Analysis: {result3['analysis'][:100]}...")

# Example 4: Nested parallel execution
print("\n4️⃣  Nested Parallel:")
print("-" * 80)

# Inner parallel for technical aspects
technical_parallel = RunnableParallel(
    architecture=PromptTemplate.from_template("Explain {topic} architecture:") | llm | parser,
    implementation=PromptTemplate.from_template("How to implement {topic}:") | llm | parser
)

# Inner parallel for business aspects
business_parallel = RunnableParallel(
    cost=PromptTemplate.from_template("Cost considerations for {topic}:") | llm | parser,
    roi=PromptTemplate.from_template("ROI of {topic}:") | llm | parser
)

# Outer parallel combining both
full_analysis = RunnableParallel(
    technical=technical_parallel,
    business=business_parallel
)

result4 = full_analysis.invoke({"topic": "Cloud Computing"})
print(f"\n📐 Technical Architecture: {result4['technical']['architecture'][:80]}...")
print(f"⚙️  Technical Implementation: {result4['technical']['implementation'][:80]}...")
print(f"💰 Business Cost: {result4['business']['cost'][:80]}...")
print(f"📈 Business ROI: {result4['business']['roi'][:80]}...")

# Example 5: Parallel then sequential
print("\n5️⃣  Parallel → Sequential:")
print("-" * 80)

def combine_analyses(parallel_result: dict) -> str:
    """Combine parallel results into one report"""
    return f"""
COMPREHENSIVE ANALYSIS:

Summary: {parallel_result['summary']}

Benefits:
{parallel_result['benefits']}

Challenges:
{parallel_result['challenges']}
"""

combined_chain = parallel | RunnableLambda(combine_analyses)
result5 = combined_chain.invoke({"topic": "Machine Learning"})
print(result5)

print("\n" + "=" * 80)
print("RUNNABLE PARALLEL EXPLAINED")
print("=" * 80)
print("""
RunnableParallel executes multiple chains concurrently:

Syntax:
    parallel = RunnableParallel(
        key1=chain1,
        key2=chain2,
        key3=chain3
    )
    
    result = parallel.invoke(input)
    # Returns: {"key1": result1, "key2": result2, "key3": result3}

Shorthand:
    parallel = {
        "key1": chain1,
        "key2": chain2
    }

Flow:
           ┌─── chain1 ───┐
    Input ─┼─── chain2 ───┼─→ {"key1": r1, "key2": r2, "key3": r3}
           └─── chain3 ───┘

Use Cases:
✅ Multiple analyses simultaneously
✅ Different perspectives on same input
✅ Parallel data extraction
✅ Redundancy/fallbacks
✅ Performance optimization

Benefits:
✅ Faster execution (concurrent)
✅ Organized outputs (keyed dict)
✅ Composable (can chain further)
✅ All inputs shared across branches
✅ Automatic error handling per branch

Nesting:
    RunnableParallel(
        branch1=RunnableParallel(...),
        branch2=RunnableParallel(...)
    )
    Creates nested parallel execution!
""")
print("=" * 80)
