"""
Runnable Basic #3: RunnableLambda (Custom Functions)
Use custom Python functions in chains
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

print("=" * 80)
print("RUNNABLE LAMBDA - CUSTOM FUNCTIONS")
print("=" * 80)

# Initialize components
llm = OllamaLLM(model="mistral")
parser = StrOutputParser()

# Define custom functions
def uppercase_input(input_dict: dict) -> dict:
    """Convert topic to uppercase"""
    print(f"  📝 Preprocessing: {input_dict['topic']} → {input_dict['topic'].upper()}")
    return {"topic": input_dict["topic"].upper()}

def add_emoji(text: str) -> str:
    """Add emoji to output"""
    result = f"✨ {text} ✨"
    print(f"  🎨 Adding emojis")
    return result

def count_words(text: str) -> dict:
    """Count words and add metadata"""
    word_count = len(text.split())
    print(f"  🔢 Counting words: {word_count}")
    return {
        "text": text,
        "word_count": word_count,
        "char_count": len(text)
    }

# Create runnables from functions
preprocess = RunnableLambda(uppercase_input)
add_emoji_runnable = RunnableLambda(add_emoji)
counter = RunnableLambda(count_words)

# Example 1: Simple preprocessing chain
print("\n1️⃣  Preprocessing Chain:")
print("-" * 80)

prompt = PromptTemplate.from_template("Explain {topic} in one sentence:")
chain1 = preprocess | prompt | llm | parser

result1 = chain1.invoke({"topic": "python"})
print(f"\nResult: {result1}\n")

# Example 2: Postprocessing chain
print("\n2️⃣  Postprocessing Chain:")
print("-" * 80)

chain2 = prompt | llm | parser | add_emoji_runnable

result2 = chain2.invoke({"topic": "JavaScript"})
print(f"\nResult: {result2}\n")

# Example 3: Multiple transformations
print("\n3️⃣  Multiple Transformations:")
print("-" * 80)

def clean_text(text: str) -> str:
    """Remove extra whitespace"""
    print(f"  🧹 Cleaning text")
    return " ".join(text.split())

def add_timestamp(text: str) -> str:
    """Add timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  ⏰ Adding timestamp")
    return f"[{timestamp}] {text}"

chain3 = (
    prompt 
    | llm 
    | parser 
    | RunnableLambda(clean_text)
    | RunnableLambda(add_timestamp)
    | add_emoji_runnable
)

result3 = chain3.invoke({"topic": "AI"})
print(f"\nResult: {result3}\n")

# Example 4: Data extraction
print("\n4️⃣  Data Extraction Chain:")
print("-" * 80)

chain4 = prompt | llm | parser | counter

result4 = chain4.invoke({"topic": "Machine Learning"})
print(f"\nText: {result4['text'][:100]}...")
print(f"Word Count: {result4['word_count']}")
print(f"Character Count: {result4['char_count']}\n")

# Example 5: Conditional logic
print("\n5️⃣  Conditional Logic:")
print("-" * 80)

def route_by_length(text: str) -> str:
    """Add different emojis based on text length"""
    length = len(text.split())
    print(f"  🔍 Text length: {length} words")
    
    if length < 20:
        return f"📝 (Short) {text}"
    elif length < 50:
        return f"📄 (Medium) {text}"
    else:
        return f"📚 (Long) {text}"

chain5 = prompt | llm | parser | RunnableLambda(route_by_length)

for topic in ["AI", "Machine Learning Benefits"]:
    result = chain5.invoke({"topic": topic})
    print(f"\n{topic}: {result[:150]}...")

print("\n" + "=" * 80)
print("RUNNABLE LAMBDA EXPLAINED")
print("=" * 80)
print("""
RunnableLambda converts Python functions into Runnables:

    def my_function(input):
        # Your logic here
        return output
    
    my_runnable = RunnableLambda(my_function)
    
    # Use in chains
    chain = component1 | my_runnable | component2

Function Requirements:
✅ Takes one input parameter
✅ Returns one output
✅ Can be sync or async
✅ Pure or with side effects

Use Cases:
- Data preprocessing
- Data postprocessing
- Custom transformations
- Validation logic
- Format conversion
- Business logic
- Filtering
- Routing

Benefits:
✅ Full Python flexibility
✅ Easy to test
✅ Reusable
✅ Type-safe (with hints)
✅ Works with any LangChain component
""")
print("=" * 80)
