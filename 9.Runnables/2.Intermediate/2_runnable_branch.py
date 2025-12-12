"""
Runnable Intermediate #2: RunnableBranch
Conditional routing based on input
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

print("=" * 80)
print("RUNNABLE BRANCH - CONDITIONAL ROUTING")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Example 1: Simple conditional routing
print("\n1️⃣  Simple Conditional Branch:")
print("-" * 80)

# Define condition functions
def is_python_question(input_dict: dict) -> bool:
    """Check if question is about Python"""
    question = input_dict["question"].lower()
    return "python" in question

def is_javascript_question(input_dict: dict) -> bool:
    """Check if question is about JavaScript"""
    question = input_dict["question"].lower()
    return "javascript" in question or "js" in question

# Define specialized chains
python_chain = (
    PromptTemplate.from_template("As a Python expert: {question}")
    | llm
    | parser
)

javascript_chain = (
    PromptTemplate.from_template("As a JavaScript expert: {question}")
    | llm
    | parser
)

general_chain = (
    PromptTemplate.from_template("Answer this question: {question}")
    | llm
    | parser
)

# Create branch
branch = RunnableBranch(
    (is_python_question, python_chain),
    (is_javascript_question, javascript_chain),
    general_chain  # default branch
)

# Test different questions
questions = [
    "How do I use list comprehensions in Python?",
    "What are JavaScript promises?",
    "What is a database?"
]

for q in questions:
    print(f"\n❓ Question: {q}")
    result = branch.invoke({"question": q})
    print(f"📝 Answer: {result[:150]}...\n")
    print("-" * 40)

# Example 2: Priority-based routing
print("\n2️⃣  Priority-Based Routing:")
print("-" * 80)

def is_urgent(input_dict: dict) -> bool:
    """Check if query is urgent"""
    query = input_dict["query"].lower()
    urgent_keywords = ["urgent", "emergency", "critical", "asap"]
    return any(keyword in query for keyword in urgent_keywords)

def is_technical(input_dict: dict) -> bool:
    """Check if query is technical"""
    query = input_dict["query"].lower()
    tech_keywords = ["code", "bug", "error", "api", "database"]
    return any(keyword in query for keyword in tech_keywords)

urgent_chain = (
    PromptTemplate.from_template("URGENT REQUEST: {query}\nProvide immediate solution:")
    | llm
    | parser
    | RunnableLambda(lambda x: f"🚨 URGENT: {x}")
)

technical_chain = (
    PromptTemplate.from_template("Technical query: {query}\nProvide detailed technical answer:")
    | llm
    | parser
    | RunnableLambda(lambda x: f"🔧 TECHNICAL: {x}")
)

standard_chain = (
    PromptTemplate.from_template("Query: {query}")
    | llm
    | parser
    | RunnableLambda(lambda x: f"💬 STANDARD: {x}")
)

priority_branch = RunnableBranch(
    (is_urgent, urgent_chain),      # Check urgent first
    (is_technical, technical_chain),  # Then technical
    standard_chain                    # Default
)

test_queries = [
    "What is Python?",
    "I have a critical bug in my API code!",
    "How do I fix this database error?"
]

for query in test_queries:
    print(f"\n📥 Query: {query}")
    result = priority_branch.invoke({"query": query})
    print(f"📤 Response: {result[:100]}...\n")

# Example 3: Data type routing
print("\n3️⃣  Data Type Routing:")
print("-" * 80)

def is_number_input(input_dict: dict) -> bool:
    """Check if input is a number"""
    try:
        float(input_dict["value"])
        return True
    except ValueError:
        return False

def is_list_input(input_dict: dict) -> bool:
    """Check if input looks like a list"""
    value = str(input_dict["value"])
    return value.startswith("[") and value.endswith("]")

number_processor = RunnableLambda(
    lambda x: f"Numeric value: {float(x['value'])} (squared: {float(x['value'])**2})"
)

list_processor = RunnableLambda(
    lambda x: f"List detected: {x['value']} (length: {len(eval(x['value']))})"
)

string_processor = RunnableLambda(
    lambda x: f"String value: '{x['value']}' (length: {len(x['value'])} chars)"
)

type_router = RunnableBranch(
    (is_number_input, number_processor),
    (is_list_input, list_processor),
    string_processor
)

test_values = ["42", "[1, 2, 3, 4]", "Hello World"]

for value in test_values:
    result = type_router.invoke({"value": value})
    print(f"{value:20} → {result}")

# Example 4: Multi-level branching
print("\n\n4️⃣  Multi-Level Branching:")
print("-" * 80)

def is_programming_topic(input_dict: dict) -> bool:
    topic = input_dict["topic"].lower()
    return any(lang in topic for lang in ["python", "javascript", "java", "programming"])

def is_data_science_topic(input_dict: dict) -> bool:
    topic = input_dict["topic"].lower()
    return any(term in topic for term in ["data", "machine learning", "ai", "analytics"])

# Sub-branch for programming
def is_web_dev(input_dict: dict) -> bool:
    topic = input_dict["topic"].lower()
    return any(term in topic for term in ["web", "html", "css", "react"])

web_chain = PromptTemplate.from_template("Web development topic: {topic}") | llm | parser
backend_chain = PromptTemplate.from_template("Backend development topic: {topic}") | llm | parser

programming_subbranch = RunnableBranch(
    (is_web_dev, web_chain),
    backend_chain  # default for programming
)

# Main branch
data_science_chain = PromptTemplate.from_template("Data science topic: {topic}") | llm | parser
general_topic_chain = PromptTemplate.from_template("General topic: {topic}") | llm | parser

main_branch = RunnableBranch(
    (is_programming_topic, programming_subbranch),
    (is_data_science_topic, data_science_chain),
    general_topic_chain
)

topics = ["Python web development", "Machine Learning", "History"]
for topic in topics:
    print(f"\n📚 Topic: {topic}")
    result = main_branch.invoke({"topic": topic})
    print(f"💡 Response: {result[:100]}...")

print("\n" + "=" * 80)
print("RUNNABLE BRANCH EXPLAINED")
print("=" * 80)
print("""
RunnableBranch implements if-then-else logic:

Syntax:
    branch = RunnableBranch(
        (condition1, chain1),  # if condition1 → run chain1
        (condition2, chain2),  # elif condition2 → run chain2
        default_chain          # else → run default_chain
    )

Condition function:
    def my_condition(input: dict) -> bool:
        return True  # or False

Flow:
    Input → Condition1? ─Yes→ Chain1 → Output
              │
              No
              ↓
            Condition2? ─Yes→ Chain2 → Output
              │
              No
              ↓
            DefaultChain → Output

Features:
✅ Evaluated top to bottom
✅ First matching condition wins
✅ Default branch is required
✅ Can nest branches
✅ Conditions are Python functions

Use Cases:
- Route by content type
- Language detection
- Priority handling
- Data type routing
- Business rules
- Feature flags
- A/B testing

Nesting:
    RunnableBranch(
        (cond1, RunnableBranch(...)),  # Nested branch
        (cond2, chain2),
        default
    )
    Creates decision trees!

Comparison with RouterChain:
- RunnableBranch: Deterministic Python conditions
- RouterChain: LLM-based routing (slower, more flexible)
""")
print("=" * 80)
