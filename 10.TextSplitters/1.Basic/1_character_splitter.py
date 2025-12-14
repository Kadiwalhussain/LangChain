"""
Basic Text Splitter #1: Character Text Splitter
Split text by character count with simple separator logic
"""

from langchain_text_splitters import CharacterTextSplitter

print("=" * 80)
print("CHARACTER TEXT SPLITTER - BASIC")
print("=" * 80)

# Sample text
sample_text = """
LangChain is a framework for developing applications powered by language models.
It enables applications that are context-aware and reason.

The framework consists of several parts:
- LangChain Libraries: Python and JavaScript libraries
- LangChain Templates: A collection of reference architectures
- LangServe: Deploy LangChain chains as REST APIs
- LangSmith: A developer platform for debugging and monitoring

LangChain simplifies every stage of the LLM application lifecycle:
Development: Build applications using LangChain's building blocks
Productionization: Use LangSmith to inspect, monitor and evaluate chains
Deployment: Turn any chain into an API with LangServe
"""

print("\n📄 Original Text:")
print(f"   Length: {len(sample_text)} characters")
print(f"   Preview: {sample_text[:100]}...")

# Example 1: Simple character splitter
print("\n" + "=" * 80)
print("Example 1: Basic Character Splitting")
print("=" * 80)

splitter1 = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)

chunks1 = splitter1.split_text(sample_text)

print(f"\n✅ Created {len(chunks1)} chunks")
for i, chunk in enumerate(chunks1, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(f"   {chunk[:80]}...")

# Example 2: Different separator
print("\n" + "=" * 80)
print("Example 2: Sentence-based Splitting")
print("=" * 80)

splitter2 = CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separator="."
)

chunks2 = splitter2.split_text(sample_text)

print(f"\n✅ Created {len(chunks2)} chunks")
for i, chunk in enumerate(chunks2[:3], 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(f"   {chunk.strip()}")

# Example 3: No overlap
print("\n" + "=" * 80)
print("Example 3: No Overlap (Compare)")
print("=" * 80)

splitter3 = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator="\n"
)

chunks3 = splitter3.split_text(sample_text)

print(f"\n✅ Created {len(chunks3)} chunks")
print("\n⚠️  Warning: No overlap means context can be lost between chunks!")
print(f"   Last 30 chars of Chunk 1: ...{chunks3[0][-30:]}")
print(f"   First 30 chars of Chunk 2: {chunks3[1][:30]}...")

# Example 4: Document splitting
print("\n" + "=" * 80)
print("Example 4: Splitting Documents")
print("=" * 80)

from langchain_core.documents import Document

# Create documents
docs = [
    Document(page_content="Document 1 content here.", metadata={"source": "doc1.txt"}),
    Document(page_content="Document 2 content here.", metadata={"source": "doc2.txt"}),
]

split_docs = splitter1.split_documents(docs)

print(f"\n✅ Split {len(docs)} documents into {len(split_docs)} chunks")
for doc in split_docs:
    print(f"   Content: {doc.page_content[:50]}... | Source: {doc.metadata['source']}")

# Visual demonstration
print("\n" + "=" * 80)
print("VISUALIZATION: How Character Splitter Works")
print("=" * 80)

demo_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
demo_splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=3,
    separator=""
)

demo_chunks = demo_splitter.split_text(demo_text)

print(f"\nOriginal: {demo_text}")
print(f"Chunk size: 10, Overlap: 3\n")

for i, chunk in enumerate(demo_chunks, 1):
    print(f"Chunk {i}: [{chunk}]")

print("\n" + "=" * 80)
print("KEY PARAMETERS")
print("=" * 80)
print("""
1. chunk_size: Maximum characters per chunk (e.g., 1000)
2. chunk_overlap: Characters shared between chunks (e.g., 200)
3. separator: Character to split on (e.g., "\\n", ".", " ")

Usage:
  splitter = CharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      separator="\\n"
  )
  chunks = splitter.split_text(text)
""")

print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Simple and fast
   - Predictable behavior
   - Good for uniform text

❌ Cons:
   - May break mid-sentence
   - Doesn't respect document structure
   - Less intelligent than recursive splitter

💡 Recommendation: Use RecursiveCharacterTextSplitter for better results
""")
