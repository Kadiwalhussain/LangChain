"""
Intermediate Chain #2: Router Chain
Route inputs to different chains based on conditions
"""

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

print("=" * 80)
print("ROUTER CHAIN - CONDITIONAL ROUTING")
print("=" * 80)

# Initialize LLM
llm = OllamaLLM(model="mistral")

# Define different destination chains
physics_template = """You are a physics expert. Answer this physics question:
{input}

Physics Answer:"""

math_template = """You are a math expert. Solve this math problem step by step:
{input}

Math Solution:"""

programming_template = """You are a programming expert. Help with this coding question:
{input}

Programming Help:"""

# Create destination chains
physics_prompt = PromptTemplate(template=physics_template, input_variables=["input"])
math_prompt = PromptTemplate(template=math_template, input_variables=["input"])
programming_prompt = PromptTemplate(template=programming_template, input_variables=["input"])

destination_chains = {
    "physics": LLMChain(llm=llm, prompt=physics_prompt),
    "math": LLMChain(llm=llm, prompt=math_prompt),
    "programming": LLMChain(llm=llm, prompt=programming_prompt)
}

# Default chain
default_prompt = PromptTemplate(
    template="Answer this general question:\n{input}",
    input_variables=["input"]
)
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Router prompt
destinations = [
    "physics: Questions about physics, forces, energy, mechanics",
    "math: Mathematical problems, calculations, equations",
    "programming: Code questions, algorithms, programming concepts"
]

destinations_str = "\n".join(destinations)
router_template = f"""Given a user input, select the most appropriate expert.

{destinations_str}

<< INPUT >>
{{input}}

<< OUTPUT (must be a single word: physics, math, programming, or DEFAULT) >>
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"]
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Create the overall router chain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# Test with different types of questions
questions = [
    "What is Newton's second law?",
    "Calculate the integral of x^2",
    "How do I write a for loop in Python?",
    "What is the meaning of life?"
]

print("\n🔀 Testing Router Chain:\n")

for i, question in enumerate(questions, 1):
    print(f"\n{i}. Question: {question}")
    try:
        result = chain.invoke({"input": question})
        print(f"   Answer: {result['text'][:150]}...")
    except Exception as e:
        print(f"   Error: {e}")

print("\n" + "=" * 80)
print("ROUTER CHAIN FLOW")
print("=" * 80)
print("""
                    Input Question
                          ↓
                   Router (LLM)
                          ↓
        ┌─────────────────┼─────────────────┬─────────┐
        ↓                 ↓                 ↓         ↓
    Physics          Math            Programming   Default
      Chain          Chain              Chain       Chain
        ↓                 ↓                 ↓         ↓
                    Appropriate Answer

Benefits:
✅ Specialized responses
✅ Better accuracy
✅ Domain-specific handling
✅ Automatic routing
""")
print("=" * 80)
