"""
Advanced Text Splitter #1: Semantic Chunker
AI-powered splitting based on semantic similarity
"""

print("=" * 80)
print("SEMANTIC CHUNKER - AI-POWERED SPLITTING")
print("=" * 80)

print("""
⚠️  Note: Semantic Chunker requires:
   - OpenAI API key (or other embedding provider)
   - langchain-experimental package
   - Active internet connection

This is the most advanced splitting method that uses embeddings
to determine natural semantic boundaries in text.
""")

# Example 1: Concept Overview
print("\n" + "=" * 80)
print("Example 1: How Semantic Chunking Works")
print("=" * 80)

print("""
Traditional Splitters:
- Split by character count
- Split by tokens
- Split by separators (\\n, ., etc.)
- Don't understand content meaning

Semantic Chunker:
- Generates embeddings for sentences
- Calculates similarity between adjacent sentences
- Identifies topic boundaries
- Splits when semantic similarity drops
- Creates coherent chunks based on meaning

Process:
1. Split text into sentences
2. Generate embedding for each sentence
3. Calculate similarity: sentence[i] vs sentence[i+1]
4. When similarity drops significantly → create boundary
5. Group sentences into semantically coherent chunks

Benefits:
✅ Chunks have consistent topics
✅ Better for RAG systems
✅ Improved retrieval accuracy
✅ Natural topic boundaries
✅ No arbitrary mid-topic cuts
""")

# Example 2: Code demonstration (will work with API key)
print("\n" + "=" * 80)
print("Example 2: Semantic Chunker Implementation")
print("=" * 80)

print("""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create semantic chunker
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"  # or "standard_deviation", "interquartile"
)

# Split text
text = \"\"\"
First topic content here about machine learning...

Second topic content here about data science...

Third topic content here about neural networks...
\"\"\"

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
""")

# Example 3: Comparison with other splitters
print("\n" + "=" * 80)
print("Example 3: Semantic vs Traditional Splitting")
print("=" * 80)

sample_text = """
Machine learning is a subset of artificial intelligence. It focuses on 
developing algorithms that can learn from data. Neural networks are a key 
component of modern machine learning systems.

Climate change is affecting global weather patterns. Rising temperatures 
lead to more extreme weather events. Scientists are studying these changes 
to predict future impacts.

Python is a popular programming language. It's widely used in data science 
and web development. The language emphasizes code readability and simplicity.
"""

print("Sample text with 3 distinct topics:")
print(sample_text)

print("\n📊 Character Splitter (arbitrary boundaries):")
print("   Might split: 'Neural networks are a key | component of modern...'")
print("   Problem: Breaks mid-sentence or mid-topic")

print("\n📊 Recursive Splitter (better but not semantic):")
print("   Splits by: paragraphs → sentences → words")
print("   Problem: May group unrelated paragraphs if they're short")

print("\n📊 Semantic Chunker (ideal):")
print("   Chunk 1: All ML content (sentences 1-3)")
print("   Chunk 2: All climate content (sentences 4-6)")
print("   Chunk 3: All Python content (sentences 7-9)")
print("   ✅ Each chunk is semantically coherent!")

# Example 4: Breakpoint threshold types
print("\n" + "=" * 80)
print("Example 4: Breakpoint Threshold Types")
print("=" * 80)

print("""
Three methods to determine where to split:

1. Percentile (Recommended):
   - Splits at points where similarity drops below Nth percentile
   - Default: 95th percentile
   - More splits with lower percentile (e.g., 80)
   - Fewer splits with higher percentile (e.g., 95)
   
   splitter = SemanticChunker(
       embeddings=embeddings,
       breakpoint_threshold_type="percentile",
       breakpoint_threshold_amount=95
   )

2. Standard Deviation:
   - Splits when similarity drops > N standard deviations below mean
   - More sensitive to outliers
   - Default: 3 standard deviations
   
   splitter = SemanticChunker(
       embeddings=embeddings,
       breakpoint_threshold_type="standard_deviation",
       breakpoint_threshold_amount=3
   )

3. Interquartile:
   - Uses IQR method (Q3 - Q1)
   - Splits at Q1 - N*IQR
   - Robust to outliers
   - Default: 1.5
   
   splitter = SemanticChunker(
       embeddings=embeddings,
       breakpoint_threshold_type="interquartile",
       breakpoint_threshold_amount=1.5
   )

💡 Start with "percentile" at 95 for most use cases
""")

# Example 5: Use case scenarios
print("\n" + "=" * 80)
print("Example 5: When to Use Semantic Chunker")
print("=" * 80)

print("""
✅ Best for:

1. RAG Systems:
   - More relevant chunk retrieval
   - Better context matching
   - Improved answer quality

2. Multi-topic Documents:
   - Research papers with multiple sections
   - News articles covering different aspects
   - Books with varied content

3. Knowledge Base:
   - Each chunk is a complete thought
   - Better for Q&A systems
   - Improved semantic search

4. Summarization:
   - Each chunk is self-contained
   - No mid-topic cuts
   - Better summary coherence

❌ Not ideal for:

1. Structured Content:
   - Use MarkdownTextSplitter for docs
   - Use HTMLHeaderTextSplitter for web pages
   - Use LanguageTextSplitter for code

2. Time-sensitive Applications:
   - Slower than traditional splitters
   - Requires API calls for embeddings

3. Offline Systems:
   - Needs internet connection
   - Requires embedding API access

4. Cost-sensitive Projects:
   - Embedding generation costs money
   - Higher processing costs
""")

# Example 6: Performance considerations
print("\n" + "=" * 80)
print("Example 6: Performance & Cost Analysis")
print("=" * 80)

print("""
⚡ Speed Comparison:

Character Splitter:     0.1ms per document
Recursive Splitter:     0.5ms per document
Token Splitter:         2ms per document
Semantic Chunker:       100-500ms per document (API latency)

💰 Cost Analysis (OpenAI Embeddings):

Document: 1000 words ≈ 1333 tokens
Sentences: ~50 sentences
Embeddings needed: 50 calls

Example Cost (OpenAI text-embedding-3-small): ~$0.0013 per document

For 1000 documents: ~$1.30
For 10,000 documents: ~$13.00

⚠️  Note: Pricing varies by provider and changes frequently.
   Always check current pricing at your provider's website.

💡 Optimization Tips:

1. Cache Results:
   - Store chunks in database
   - Reuse for same documents
   - Save on repeated processing

2. Batch Processing:
   - Process multiple documents at once
   - Amortize API call overhead

3. Hybrid Approach:
   - Use semantic chunker for important docs
   - Use traditional splitters for routine content

4. Local Embeddings:
   - Use HuggingFace models locally
   - No API costs
   - Slightly lower quality but free
""")

# Example 7: Alternative implementations
print("\n" + "=" * 80)
print("Example 7: Using Local Embeddings (Cost-Free)")
print("=" * 80)

print("""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Use free, local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create semantic chunker
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)

# Split text (no API costs!)
chunks = splitter.split_text(text)

Benefits:
✅ No API costs
✅ Works offline
✅ Faster (no network latency)
✅ Privacy (data stays local)

Trade-offs:
❌ Slightly lower quality than OpenAI
❌ Requires more RAM
❌ Need to download model first (~90MB)
""")

# Example 8: Configuration guide
print("\n" + "=" * 80)
print("CONFIGURATION GUIDE")
print("=" * 80)

print("""
Basic Setup:

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Method 1: OpenAI Embeddings (best quality)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = SemanticChunker(embeddings=embeddings)

# Method 2: Local Embeddings (free)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
splitter = SemanticChunker(embeddings=embeddings)

Advanced Configuration:

# Fine-tune splitting sensitivity
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,  # Lower = more splits
    number_of_chunks=None,  # Or specify exact number
    sentence_split_regex=r"(?<=[.!?])\s+"  # Custom sentence detection
)

Usage:

# Split text
chunks = splitter.split_text(text)

# Create documents with metadata
docs = splitter.create_documents([text], metadatas=[{"source": "doc1"}])
""")

# Example 9: RAG integration
print("\n" + "=" * 80)
print("Example 9: Complete RAG Pipeline with Semantic Chunker")
print("=" * 80)

print("""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Load document
with open("document.txt") as f:
    text = f.read()

# 2. Split semantically
embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(embeddings=embeddings)
chunks = splitter.split_text(text)

print(f"Created {len(chunks)} semantic chunks")

# 3. Create vector store
vectorstore = FAISS.from_texts(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create QA chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 6. Query
question = "What are the main topics discussed?"
answer = qa_chain.invoke({"query": question})
print(answer)

Benefits of Semantic Chunking in RAG:
✅ More relevant chunk retrieval
✅ Better context for answers
✅ Fewer irrelevant chunks returned
✅ Improved answer accuracy
""")

# Pros and cons
print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Most intelligent splitting
   - Preserves semantic coherence
   - Best for RAG systems
   - Natural topic boundaries
   - Improves retrieval quality
   - Self-contained chunks

❌ Cons:
   - Slowest method (API calls)
   - Costs money (embedding API)
   - Requires internet (unless local)
   - More complex setup
   - Overkill for simple tasks

💰 Cost-Benefit:
   - Worth it for production RAG systems
   - Significant quality improvement
   - Can justify cost with better UX
   
⚡ Performance:
   - ~100-500ms per document
   - Cache results for reuse
   - Consider batch processing
""")

# Quick reference
print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print("""
# Option 1: OpenAI (best quality, paid)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(embeddings=embeddings)
chunks = splitter.split_text(text)

# Option 2: HuggingFace (free, local)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
splitter = SemanticChunker(embeddings=embeddings)
chunks = splitter.split_text(text)

# Fine-tuning
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
""")

print("\n" + "=" * 80)
print("🌟 RECOMMENDATION")
print("=" * 80)
print("""
Use Semantic Chunker when:
- Building production RAG systems
- Quality > Speed/Cost
- Multi-topic documents
- Need best retrieval accuracy

Use Traditional Splitters when:
- Cost/Speed is priority
- Simple documents
- Structured content (code, markdown, HTML)
- Prototyping
""")
