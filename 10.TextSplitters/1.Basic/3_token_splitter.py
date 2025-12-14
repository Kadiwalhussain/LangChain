"""
Basic Text Splitter #3: Token Text Splitter
Split text based on token count - important for LLM token limits
"""

from langchain_text_splitters import TokenTextSplitter

print("=" * 80)
print("TOKEN TEXT SPLITTER - FOR PRECISE TOKEN MANAGEMENT")
print("=" * 80)

# Sample text
sample_text = """
Artificial intelligence (AI) is transforming the world. Machine learning algorithms 
can now recognize patterns in vast amounts of data, enabling applications from image 
recognition to natural language processing.

Large Language Models (LLMs) like GPT-4 have demonstrated remarkable capabilities in 
understanding and generating human-like text. These models are trained on diverse 
internet text and can perform tasks ranging from translation to code generation.

The key challenge with LLMs is managing their context window. Each model has a maximum 
number of tokens it can process at once. For example, GPT-3.5-Turbo has variants with 
4K, 16K token limits, while GPT-4 can handle 8K to 128K tokens depending on the version.

Token counting is crucial because:
1. API costs are calculated per token
2. Models have hard token limits
3. Response quality depends on context size
4. Different models use different tokenization schemes

This is where token-based text splitting becomes essential for production applications.
"""

print("\n📄 Original Text:")
print(f"   Character length: {len(sample_text)} characters")

# Example 1: Basic token splitting
print("\n" + "=" * 80)
print("Example 1: Basic Token-Based Splitting")
print("=" * 80)

# Default uses tiktoken for OpenAI models
splitter1 = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks1 = splitter1.split_text(sample_text)

print(f"\n✅ Created {len(chunks1)} chunks")
print(f"\n📊 Token Analysis:")
for i, chunk in enumerate(chunks1, 1):
    print(f"\nChunk {i}:")
    print(f"   Characters: {len(chunk)}")
    preview = chunk.strip()[:80].replace("\n", " ")
    print(f"   Preview: {preview}...")

# Example 2: Understanding tokens vs characters
print("\n" + "=" * 80)
print("Example 2: Tokens vs Characters")
print("=" * 80)

demo_texts = [
    "Hello world",
    "Hello world!!!",
    "Artificial Intelligence",
    "AI",
    "supercalifragilisticexpialidocious"
]

print("\n📊 Token Count Comparison:")
print(f"{'Text':<40} {'Characters':<12} {'Estimated Tokens':<15}")
print("-" * 70)

for text in demo_texts:
    char_count = len(text)
    # Rough estimate: 1 token ≈ 4 characters for English
    estimated_tokens = char_count // 4 + 1
    print(f"{text:<40} {char_count:<12} {estimated_tokens:<15}")

print("\n💡 Note: Actual token count depends on the tokenizer used!")

# Example 3: Different chunk sizes
print("\n" + "=" * 80)
print("Example 3: Adjusting Chunk Size for Different Models")
print("=" * 80)

model_configs = {
    "GPT-3.5 Turbo": {"chunk_size": 500, "overlap": 50},
    "GPT-4": {"chunk_size": 1000, "overlap": 100},
    "GPT-4-32k": {"chunk_size": 4000, "overlap": 400},
}

short_text = "LangChain provides utilities for splitting text. " * 50

for model_name, config in model_configs.items():
    splitter = TokenTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["overlap"]
    )
    chunks = splitter.split_text(short_text)
    print(f"\n{model_name}:")
    print(f"   Chunk size: {config['chunk_size']} tokens")
    print(f"   Overlap: {config['overlap']} tokens")
    print(f"   Result: {len(chunks)} chunks")

# Example 4: Cost optimization
print("\n" + "=" * 80)
print("Example 4: Cost Optimization with Token Counting")
print("=" * 80)

document = """
This is a sample document that we want to process with an LLM.
We need to be mindful of token costs when working with APIs.
Every token sent to the API incurs a cost, so optimizing chunk
size and overlap can help reduce expenses while maintaining quality.
""" * 10  # Repeat to make it longer

# Conservative splitting (more overlap, safer but more expensive)
conservative_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=30)
conservative_chunks = conservative_splitter.split_text(document)

# Aggressive splitting (less overlap, cheaper but riskier)
aggressive_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
aggressive_chunks = aggressive_splitter.split_text(document)

print(f"\n💰 Cost Comparison:")
print(f"\nConservative (30% overlap):")
print(f"   Chunks: {len(conservative_chunks)}")
print(f"   Total tokens (estimated): {len(conservative_chunks) * 100}")

print(f"\nAggressive (10% overlap):")
print(f"   Chunks: {len(aggressive_chunks)}")
print(f"   Total tokens (estimated): {len(aggressive_chunks) * 100}")

savings = len(conservative_chunks) - len(aggressive_chunks)
print(f"\n📊 Savings: ~{savings * 100} tokens ({savings} fewer chunks)")

# Example 5: Integration with LLMs
print("\n" + "=" * 80)
print("Example 5: Practical RAG Integration")
print("=" * 80)

rag_document = """
Python is a high-level programming language known for its simplicity and readability.
It supports multiple programming paradigms including procedural, object-oriented, and
functional programming. Python's extensive standard library and vibrant ecosystem of
third-party packages make it suitable for various applications.

Web development frameworks like Django and Flask enable building robust web applications.
Data science libraries such as NumPy, Pandas, and Scikit-learn provide powerful tools
for data manipulation and machine learning. Python's popularity continues to grow across
various domains including automation, scientific computing, and artificial intelligence.
"""

# Split for RAG system
rag_splitter = TokenTextSplitter(
    chunk_size=50,  # Smaller chunks for precise retrieval
    chunk_overlap=10
)

rag_chunks = rag_splitter.split_text(rag_document)

print(f"\n✅ Document split into {len(rag_chunks)} chunks for RAG")
print("\nChunks ready for embedding:")
for i, chunk in enumerate(rag_chunks[:3], 1):  # Show first 3
    print(f"\n{i}. {chunk.strip()[:100]}...")

# Example 6: Handling different languages
print("\n" + "=" * 80)
print("Example 6: Token Considerations for Different Languages")
print("=" * 80)

print("""
Token counts vary by language:

English:    ~1 token per 4 characters
Spanish:    ~1 token per 4 characters  
French:     ~1 token per 4 characters
German:     ~1 token per 5 characters
Chinese:    ~1 token per 2 characters
Japanese:   ~1 token per 2 characters

💡 Always test with actual content in your target language!
""")

# Configuration guide
print("\n" + "=" * 80)
print("CONFIGURATION GUIDE")
print("=" * 80)
print("""
Recommended Settings by Model:

1. GPT-3.5-Turbo (4K context):
   chunk_size=500, chunk_overlap=50
   
2. GPT-3.5-Turbo-16K (16K context):
   chunk_size=2000, chunk_overlap=200
   
3. GPT-4 (8K context):
   chunk_size=1000, chunk_overlap=100
   
4. GPT-4-32K (32K context):
   chunk_size=4000, chunk_overlap=400

5. Claude 2 (100K context):
   chunk_size=10000, chunk_overlap=1000

⚠️  Important: Leave room for prompts and responses!
   If model has 4K tokens, use max 2K for chunks to leave
   space for prompts (1K) and responses (1K).

💡 Formula: chunk_size = (context_window - prompt_tokens - response_tokens) / num_chunks
""")

print("\n" + "=" * 80)
print("WHEN TO USE TOKEN SPLITTER")
print("=" * 80)
print("""
✅ Use TokenTextSplitter when:
   - Working with API token limits
   - Optimizing costs (pay per token)
   - Need precise token count control
   - Building production RAG systems
   - Compliance with model constraints

❌ Don't use when:
   - Character count is sufficient
   - Speed is critical (tokenization is slower)
   - Working offline (needs tokenizer)

🌟 Best Practice: Use for production systems with API-based LLMs
""")

print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Precise token counting
   - Respects model limits
   - Better cost control
   - Prevents context overflow
   - Production-ready

❌ Cons:
   - Slower than character splitters
   - Requires tokenizer library
   - May need model-specific tokenizer
   - More complex setup

💰 Cost Impact: Can save 20-30% on API costs with proper tuning
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print("""
from langchain_text_splitters import TokenTextSplitter

# Basic usage
splitter = TokenTextSplitter(
    chunk_size=500,         # Max tokens per chunk
    chunk_overlap=50        # Tokens shared between chunks
)

chunks = splitter.split_text(text)

# Advanced: Specify encoding
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4/GPT-3.5-turbo encoding
    chunk_size=500,
    chunk_overlap=50
)

# Common encodings:
# - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-3-*
# - p50k_base: Legacy GPT-3 models (text-davinci-002, text-davinci-003)
# - r50k_base: Older GPT-3 models (davinci, curie, babbage, ada)
""")
