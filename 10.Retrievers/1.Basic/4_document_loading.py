"""
Document Loading - Basic Example
=================================

Before you can retrieve documents, you need to load them!
This example shows how to load documents from various sources.

Document Loaders in LangChain:
- Text files
- PDFs
- Web pages
- CSV/JSON
- And many more!
"""

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
import os

# ============================================================================
# Manual Document Creation
# ============================================================================

def create_documents_manually():
    """Create documents manually - the simplest approach."""
    
    print("="*80)
    print("MANUAL DOCUMENT CREATION")
    print("="*80)
    
    # Create a single document
    doc1 = Document(
        page_content="LangChain is a framework for building LLM applications.",
        metadata={"source": "manual", "author": "Me", "date": "2024"}
    )
    
    print("\n📄 Single Document:")
    print(f"   Content: {doc1.page_content}")
    print(f"   Metadata: {doc1.metadata}")
    
    # Create multiple documents
    documents = [
        Document(
            page_content="Python is a programming language.",
            metadata={"topic": "programming", "difficulty": "beginner"}
        ),
        Document(
            page_content="Machine learning models learn from data.",
            metadata={"topic": "AI", "difficulty": "intermediate"}
        ),
        Document(
            page_content="Neural networks have multiple layers.",
            metadata={"topic": "AI", "difficulty": "advanced"}
        )
    ]
    
    print(f"\n📚 Created {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(f"\n   {i}. {doc.page_content}")
        print(f"      Metadata: {doc.metadata}")
    
    return documents


# ============================================================================
# Text Splitting
# ============================================================================

def demonstrate_text_splitting():
    """Show how to split large texts into chunks."""
    
    print("\n\n" + "="*80)
    print("TEXT SPLITTING")
    print("="*80)
    
    # Long text that needs splitting
    long_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications to be context-aware and reason through problems.
    
    The framework consists of several key components. First, there are LLMs and Chat Models
    which provide the core language understanding capabilities. Second, there are Prompts
    which are templates for formatting input to models.
    
    Third, there are Chains which combine multiple components into workflows. Fourth, there
    are Agents which can dynamically decide what actions to take. Finally, there is Memory
    which allows systems to maintain state across interactions.
    
    LangChain also provides integrations with many tools and services. These include vector
    stores for semantic search, document loaders for various file formats, and output parsers
    for structuring model responses.
    """
    
    print("\n📝 Original text:")
    print(f"   Length: {len(long_text)} characters")
    print(f"   Preview: {long_text[:100].strip()}...")
    
    # Splitter 1: CharacterTextSplitter
    print("\n" + "-"*80)
    print("SPLITTER 1: CharacterTextSplitter")
    print("-"*80)
    
    char_splitter = CharacterTextSplitter(
        separator="\n\n",  # Split on double newlines (paragraphs)
        chunk_size=200,     # Target size
        chunk_overlap=20    # Overlap between chunks
    )
    
    char_chunks = char_splitter.split_text(long_text)
    
    print(f"\nSplit into {len(char_chunks)} chunks:")
    for i, chunk in enumerate(char_chunks, 1):
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk.strip()[:100]}...")
    
    # Splitter 2: RecursiveCharacterTextSplitter (RECOMMENDED)
    print("\n" + "-"*80)
    print("SPLITTER 2: RecursiveCharacterTextSplitter (RECOMMENDED)")
    print("-"*80)
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]  # Try these in order
    )
    
    recursive_chunks = recursive_splitter.split_text(long_text)
    
    print(f"\nSplit into {len(recursive_chunks)} chunks:")
    for i, chunk in enumerate(recursive_chunks, 1):
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk.strip()[:100]}...")
    
    # Create documents from chunks
    print("\n" + "-"*80)
    print("CREATING DOCUMENTS FROM CHUNKS")
    print("-"*80)
    
    documents = recursive_splitter.create_documents(
        [long_text],
        metadatas=[{"source": "langchain_guide", "version": "1.0"}]
    )
    
    print(f"\nCreated {len(documents)} Document objects:")
    for i, doc in enumerate(documents, 1):
        print(f"\n   Document {i}:")
        print(f"   Content: {doc.page_content[:80].strip()}...")
        print(f"   Metadata: {doc.metadata}")


# ============================================================================
# Loading from Different Sources
# ============================================================================

def load_from_text_file():
    """Load documents from text files."""
    
    print("\n\n" + "="*80)
    print("LOADING FROM TEXT FILE")
    print("="*80)
    
    # Create a sample text file
    sample_content = """
    Title: Introduction to Python
    
    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It was first released in 1991 and has become one of the most popular programming languages.
    
    Python emphasizes code readability and uses significant whitespace.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    
    Key Features:
    - Easy to learn and read
    - Large standard library
    - Cross-platform compatibility
    - Strong community support
    """
    
    # Create temporary file
    filename = "temp_python_intro.txt"
    with open(filename, 'w') as f:
        f.write(sample_content)
    
    print(f"\n✅ Created sample file: {filename}")
    
    # Read file manually
    with open(filename, 'r') as f:
        content = f.read()
    
    # Create document
    doc = Document(
        page_content=content,
        metadata={"source": filename, "type": "text"}
    )
    
    print(f"\n📄 Loaded document:")
    print(f"   Content length: {len(doc.page_content)} characters")
    print(f"   Metadata: {doc.metadata}")
    print(f"   Preview: {doc.page_content[:100].strip()}...")
    
    # Clean up
    os.remove(filename)
    print(f"\n✅ Cleaned up temporary file")


# ============================================================================
# Working with Structured Data
# ============================================================================

def load_structured_data():
    """Load and process structured data."""
    
    print("\n\n" + "="*80)
    print("LOADING STRUCTURED DATA")
    print("="*80)
    
    # Example: CSV-like data
    data_rows = [
        {"name": "Alice", "role": "Engineer", "project": "RAG System"},
        {"name": "Bob", "role": "Designer", "project": "UI/UX"},
        {"name": "Charlie", "role": "Manager", "project": "Strategy"},
    ]
    
    print("\n📊 Original structured data:")
    for row in data_rows:
        print(f"   {row}")
    
    # Convert to documents
    documents = []
    for row in data_rows:
        # Create a readable text representation
        content = f"{row['name']} is a {row['role']} working on {row['project']}"
        
        doc = Document(
            page_content=content,
            metadata=row  # Keep original data as metadata
        )
        documents.append(doc)
    
    print(f"\n📄 Converted to {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\n   {i}. Content: {doc.page_content}")
        print(f"      Metadata: {doc.metadata}")


# ============================================================================
# Best Practices for Document Loading
# ============================================================================

def document_loading_best_practices():
    """Show best practices for document loading and splitting."""
    
    print("\n\n" + "="*80)
    print("DOCUMENT LOADING BEST PRACTICES")
    print("="*80)
    
    print("""
1. CHUNK SIZE SELECTION
   
   Small chunks (200-500 chars):
   ✅ More precise retrieval
   ✅ Better for specific facts
   ❌ May lose context
   ❌ More chunks to process
   
   Medium chunks (500-1500 chars):
   ✅ Good balance
   ✅ Maintains context
   ✅ Most common choice
   
   Large chunks (1500-3000 chars):
   ✅ More context
   ❌ May include irrelevant info
   ❌ Larger context windows needed

2. CHUNK OVERLAP
   
   No overlap (0%):
   ❌ Information at chunk boundaries may be lost
   
   Small overlap (10-15%):
   ✅ Recommended
   ✅ Ensures continuity
   
   Large overlap (30%+):
   ❌ Redundant information
   ❌ More storage needed

3. METADATA BEST PRACTICES
   
   Always include:
   ✅ source: Where did this come from?
   ✅ date: When was it created/modified?
   ✅ type: What kind of document?
   
   Consider adding:
   ✅ author: Who created it?
   ✅ category: How to organize it?
   ✅ tags: What topics does it cover?

4. TEXT SPLITTING STRATEGIES
   
   For general text:
   → Use RecursiveCharacterTextSplitter
   
   For code:
   → Use language-specific splitters
   
   For structured data:
   → Split by natural boundaries (paragraphs, sections)
   
   For markdown:
   → Use MarkdownTextSplitter

5. COMMON PATTERNS
   
   Pattern 1: Load → Split → Embed
   ```
   docs = loader.load()
   chunks = splitter.split_documents(docs)
   vectorstore = Chroma.from_documents(chunks, embeddings)
   ```
   
   Pattern 2: Load → Filter → Split → Embed
   ```
   docs = loader.load()
   filtered = [d for d in docs if condition(d)]
   chunks = splitter.split_documents(filtered)
   vectorstore = Chroma.from_documents(chunks, embeddings)
   ```
   
   Pattern 3: Load → Transform → Split → Embed
   ```
   docs = loader.load()
   transformed = [transform(d) for d in docs]
   chunks = splitter.split_documents(transformed)
   vectorstore = Chroma.from_documents(chunks, embeddings)
   ```
    """)


# ============================================================================
# Recommended Configurations
# ============================================================================

def recommended_configurations():
    """Show recommended configurations for different use cases."""
    
    print("\n\n" + "="*80)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*80)
    
    configs = {
        "Q&A System": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "splitter": "RecursiveCharacterTextSplitter",
            "reasoning": "Good balance of context and precision"
        },
        "Code Documentation": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "splitter": "Language-specific splitter",
            "reasoning": "Respect code structure, smaller chunks for functions"
        },
        "Long Articles/Books": {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "splitter": "RecursiveCharacterTextSplitter",
            "reasoning": "Larger chunks to maintain narrative flow"
        },
        "Product Descriptions": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "splitter": "CharacterTextSplitter",
            "reasoning": "Short, self-contained descriptions"
        },
        "Legal Documents": {
            "chunk_size": 1200,
            "chunk_overlap": 200,
            "splitter": "RecursiveCharacterTextSplitter",
            "reasoning": "Preserve clause context, moderate chunks"
        }
    }
    
    for use_case, config in configs.items():
        print(f"\n{use_case}:")
        print(f"   Chunk Size: {config['chunk_size']}")
        print(f"   Overlap: {config['chunk_overlap']}")
        print(f"   Splitter: {config['splitter']}")
        print(f"   Why: {config['reasoning']}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("DOCUMENT LOADING - COMPLETE DEMO")
    print("🚀"*40)
    
    # Run all demos
    create_documents_manually()
    demonstrate_text_splitting()
    load_from_text_file()
    load_structured_data()
    document_loading_best_practices()
    recommended_configurations()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Documents in LangChain:
   - page_content: The text content
   - metadata: Additional information (source, date, etc.)

2. Text Splitting is Essential:
   - Large documents must be split into chunks
   - RecursiveCharacterTextSplitter is recommended
   - Use appropriate chunk size and overlap

3. Chunk Size Guidelines:
   - Small (200-500): Precise but less context
   - Medium (500-1500): Balanced (RECOMMENDED)
   - Large (1500+): More context but may include noise

4. Always Add Metadata:
   - Helps with filtering and attribution
   - Essential for multi-document systems
   - Makes debugging easier

5. Common Pipeline:
   Load → Split → Embed → Store → Retrieve

6. Test Your Configuration:
   - Different content types need different settings
   - Test retrieval quality
   - Adjust chunk size based on results
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")

