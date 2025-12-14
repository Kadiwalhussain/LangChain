"""
Basic Text Splitter #2: Recursive Character Text Splitter
Smart splitting with fallback separators - RECOMMENDED for most use cases
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

print("=" * 80)
print("RECURSIVE CHARACTER TEXT SPLITTER - RECOMMENDED")
print("=" * 80)

# Sample text with different structures
sample_text = """
# Introduction to LangChain

LangChain is a framework for developing applications powered by language models.

## Core Concepts

The framework enables applications that are context-aware and can reason about their responses. This is achieved through several key components.

### Components

1. LLMs and Chat Models: The foundation for text generation
2. Prompt Templates: Reusable prompts with variables
3. Output Parsers: Structure LLM outputs
4. Chains: Combine multiple components

## Why Use LangChain?

LangChain simplifies the development process. It provides building blocks that are:
- Modular and composable
- Easy to customize
- Production-ready

You can build sophisticated AI applications without reinventing common patterns. The framework handles the complexity while you focus on your application logic.

## Getting Started

To start using LangChain, install it via pip:
pip install langchain

Then import and use the components you need for your application.
"""

print("\n📄 Original Text:")
print(f"   Length: {len(sample_text)} characters")
print(f"   Lines: {len(sample_text.splitlines())}")

# Example 1: Default recursive splitter (BEST PRACTICE)
print("\n" + "=" * 80)
print("Example 1: Default Recursive Splitter (Recommended)")
print("=" * 80)

splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks1 = splitter1.split_text(sample_text)

print(f"\n✅ Created {len(chunks1)} chunks")
print(f"\n📊 Chunk Analysis:")
for i, chunk in enumerate(chunks1, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    preview = chunk.strip()[:100].replace("\n", " ")
    print(f"   {preview}...")

# Example 2: How recursive splitting works
print("\n" + "=" * 80)
print("Example 2: Understanding the Recursive Process")
print("=" * 80)

print("""
The splitter tries separators in order:
1. First tries: "\\n\\n" (paragraphs)
2. If too large, tries: "\\n" (lines)
3. If still too large, tries: ". " (sentences)
4. If still too large, tries: " " (words)
5. Last resort: "" (characters)

This keeps content semantically coherent!
""")

# Demonstrate with different separator priorities
demo_text = "Paragraph 1 line 1.\nParagraph 1 line 2.\n\nParagraph 2 line 1.\nParagraph 2 line 2."

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", ". ", " "]
)

demo_chunks = splitter2.split_text(demo_text)

print(f"Demo text: {demo_text[:80]}...")
print(f"\n✅ Result: {len(demo_chunks)} chunks")
for i, chunk in enumerate(demo_chunks, 1):
    print(f"\nChunk {i}: {chunk}")

# Example 3: Comparison with character splitter
print("\n" + "=" * 80)
print("Example 3: Recursive vs Character Splitter")
print("=" * 80)

from langchain_text_splitters import CharacterTextSplitter

comparison_text = """First paragraph with important information.

Second paragraph with more important information.

Third paragraph concluding the section."""

# Character splitter
char_splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separator="\n"
)
char_chunks = char_splitter.split_text(comparison_text)

# Recursive splitter
rec_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", ". ", " "]
)
rec_chunks = rec_splitter.split_text(comparison_text)

print("\n📌 Character Splitter Result:")
for i, chunk in enumerate(char_chunks, 1):
    print(f"   Chunk {i}: {chunk.strip()[:60]}...")

print("\n📌 Recursive Splitter Result:")
for i, chunk in enumerate(rec_chunks, 1):
    print(f"   Chunk {i}: {chunk.strip()[:60]}...")

print("\n💡 Notice: Recursive splitter keeps paragraphs together better!")

# Example 4: Real-world document splitting
print("\n" + "=" * 80)
print("Example 4: Real-World Use Case - RAG System")
print("=" * 80)

long_document = """
Artificial Intelligence (AI) has revolutionized many industries. Machine learning, 
a subset of AI, enables computers to learn from data without explicit programming.

Deep learning takes this further by using neural networks with multiple layers. 
These networks can recognize patterns in images, understand natural language, and 
make complex decisions.

Natural Language Processing (NLP) is another key area. It allows machines to 
understand and generate human language. Applications include chatbots, translation 
services, and sentiment analysis.

The future of AI holds exciting possibilities. From autonomous vehicles to 
personalized medicine, AI will continue to transform how we live and work.
"""

rag_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""]
)

rag_chunks = rag_splitter.split_text(long_document)

print(f"\n✅ Document split into {len(rag_chunks)} chunks for RAG")
print("\nChunks for vector embedding:")
for i, chunk in enumerate(rag_chunks, 1):
    print(f"\n{i}. {chunk.strip()}")

# Example 5: Custom separators
print("\n" + "=" * 80)
print("Example 5: Custom Separators for Specific Formats")
print("=" * 80)

# For bullet points
bullet_text = """
Key Features:
• Feature one with details
• Feature two with more information
• Feature three with additional context
• Feature four wrapping up
"""

bullet_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["•", "\n", " "]
)

bullet_chunks = bullet_splitter.split_text(bullet_text)

print(f"\n✅ Bullet point text split into {len(bullet_chunks)} chunks")
for i, chunk in enumerate(bullet_chunks, 1):
    print(f"\nChunk {i}: {chunk.strip()}")

# Configuration tips
print("\n" + "=" * 80)
print("CONFIGURATION GUIDE")
print("=" * 80)
print("""
Recommended Settings by Use Case:

1. RAG Systems (Vector Search):
   chunk_size=1000, chunk_overlap=200
   separators=["\\n\\n", "\\n", ". ", " ", ""]

2. Small Context Models:
   chunk_size=500, chunk_overlap=100
   separators=["\\n\\n", "\\n", ". ", " ", ""]

3. Summarization:
   chunk_size=2000, chunk_overlap=200
   separators=["\\n\\n", "\\n", ". "]

4. Fine-tuning Data:
   chunk_size=512, chunk_overlap=50
   separators=["\\n\\n", "\\n"]

💡 General Rule: chunk_overlap = 10-20% of chunk_size
""")

print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Maintains semantic coherence
   - Respects document structure
   - Tries natural boundaries first
   - Works well for most content
   - Highly configurable

❌ Cons:
   - Slightly slower than CharacterTextSplitter
   - May need tuning for specific formats

🌟 BEST PRACTICE: Use this as your default splitter!
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print("""
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Max characters per chunk
    chunk_overlap=200,      # Characters shared between chunks
    separators=["\\n\\n", "\\n", ". ", " ", ""]  # Try in order
)

chunks = splitter.split_text(text)
""")
