"""
Vector Store Retriever - Basic Example
======================================

This example demonstrates the most fundamental retriever: Vector Store Retriever.
It uses semantic search to find documents similar to your query.

How it works:
1. Documents are converted to embeddings (numerical vectors)
2. Query is converted to an embedding
3. Find documents with similar embeddings (cosine similarity)
4. Return top-k most similar documents

Use Case: General semantic search, Q&A systems, document retrieval
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Option 1: Using OpenAI Embeddings (requires API key)
# ============================================================================

def vector_retriever_openai():
    """Create vector retriever with OpenAI embeddings."""
    
    print("="*80)
    print("VECTOR STORE RETRIEVER - OpenAI Embeddings")
    print("="*80)
    
    # Sample documents
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on learning from data.",
            metadata={"source": "ml_basics.txt", "topic": "machine_learning"}
        ),
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"source": "python_intro.txt", "topic": "programming"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks in animal brains.",
            metadata={"source": "neural_nets.txt", "topic": "machine_learning"}
        ),
        Document(
            page_content="Data science combines statistics, programming, and domain expertise to extract insights from data.",
            metadata={"source": "data_science.txt", "topic": "data_science"}
        ),
        Document(
            page_content="Deep learning uses multiple layers of neural networks to learn hierarchical representations.",
            metadata={"source": "deep_learning.txt", "topic": "machine_learning"}
        ),
        Document(
            page_content="JavaScript is a programming language commonly used for web development and creating interactive websites.",
            metadata={"source": "javascript.txt", "topic": "programming"}
        )
    ]
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="basic_demo"
    )
    
    print("\n✅ Created vector store with", len(documents), "documents")
    
    # Create retriever with different configurations
    print("\n" + "="*80)
    print("CONFIGURATION 1: Simple Similarity Search (k=2)")
    print("="*80)
    
    retriever_simple = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Return top 2 results
    )
    
    query = "What is machine learning?"
    print(f"\n🔍 Query: {query}")
    
    results = retriever_simple.get_relevant_documents(query)
    
    print(f"\n📄 Retrieved {len(results)} documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Topic: {doc.metadata['topic']}\n")
    
    # Configuration 2: With score threshold
    print("="*80)
    print("CONFIGURATION 2: Similarity with Score Threshold")
    print("="*80)
    
    retriever_threshold = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,  # Only return docs above this score
            "k": 4
        }
    )
    
    query2 = "Tell me about programming languages"
    print(f"\n🔍 Query: {query2}")
    
    results2 = retriever_threshold.get_relevant_documents(query2)
    
    print(f"\n📄 Retrieved {len(results2)} documents (above threshold):\n")
    for i, doc in enumerate(results2, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}\n")
    
    # Configuration 3: MMR (Maximum Marginal Relevance)
    # Balances relevance with diversity
    print("="*80)
    print("CONFIGURATION 3: MMR (Maximum Marginal Relevance)")
    print("="*80)
    print("MMR reduces redundancy by selecting diverse results")
    
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,  # Final number of results
            "fetch_k": 6,  # Initial candidates to consider
            "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
        }
    )
    
    query3 = "artificial intelligence and learning"
    print(f"\n🔍 Query: {query3}")
    
    results3 = retriever_mmr.get_relevant_documents(query3)
    
    print(f"\n📄 Retrieved {len(results3)} diverse documents:\n")
    for i, doc in enumerate(results3, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Topic: {doc.metadata['topic']}\n")
    
    # Clean up
    vectorstore.delete_collection()
    print("\n✅ Cleaned up vector store")


# ============================================================================
# Option 2: Using HuggingFace Embeddings (Free, Local)
# ============================================================================

def vector_retriever_huggingface():
    """Create vector retriever with HuggingFace embeddings (free, runs locally)."""
    
    print("\n\n" + "="*80)
    print("VECTOR STORE RETRIEVER - HuggingFace Embeddings (Local)")
    print("="*80)
    
    # Sample documents about different topics
    documents = [
        Document(
            page_content="The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was built between 1887 and 1889.",
            metadata={"category": "landmarks", "location": "Paris"}
        ),
        Document(
            page_content="Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
            metadata={"category": "programming", "language": "Python"}
        ),
        Document(
            page_content="The Great Wall of China is a series of fortifications built across northern China.",
            metadata={"category": "landmarks", "location": "China"}
        ),
        Document(
            page_content="JavaScript was created by Brendan Eich in 1995. It's primarily used for web development.",
            metadata={"category": "programming", "language": "JavaScript"}
        ),
        Document(
            page_content="The Taj Mahal is an ivory-white marble mausoleum in Agra, India, built by Mughal emperor Shah Jahan.",
            metadata={"category": "landmarks", "location": "India"}
        )
    ]
    
    # Create HuggingFace embeddings (runs locally, no API key needed)
    print("\n⏳ Loading HuggingFace embedding model (first time may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Small, fast model
    )
    print("✅ Model loaded successfully")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="huggingface_demo"
    )
    
    print(f"✅ Created vector store with {len(documents)} documents")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    # Test queries
    queries = [
        "Tell me about famous buildings",
        "What programming languages are there?",
        "Historical monuments in Asia"
    ]
    
    print("\n" + "="*80)
    print("TESTING MULTIPLE QUERIES")
    print("="*80)
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.get_relevant_documents(query)
        
        print(f"📄 Top {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n  {i}. {doc.page_content}")
            print(f"     Category: {doc.metadata['category']}")
    
    # Clean up
    vectorstore.delete_collection()
    print("\n\n✅ Demo completed and cleaned up")


# ============================================================================
# Understanding Embeddings
# ============================================================================

def demonstrate_embeddings():
    """Show how text is converted to embeddings."""
    
    print("\n\n" + "="*80)
    print("UNDERSTANDING EMBEDDINGS")
    print("="*80)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Embed some text
    texts = [
        "machine learning",
        "artificial intelligence",  # Similar to above
        "pizza recipe"  # Different topic
    ]
    
    print("\nEmbedding texts...")
    vectors = embeddings.embed_documents(texts)
    
    print(f"\nEach text is converted to a vector of {len(vectors[0])} dimensions")
    print("\nFirst 10 dimensions of each embedding:")
    
    for text, vector in zip(texts, vectors):
        print(f"\n'{text}':")
        print(f"  {vector[:10]}...")
    
    # Calculate similarity
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    
    print("\n" + "-"*80)
    print("SIMILARITY SCORES (higher = more similar):")
    print("-"*80)
    
    sim_1_2 = cosine_similarity(vectors[0], vectors[1])
    sim_1_3 = cosine_similarity(vectors[0], vectors[2])
    sim_2_3 = cosine_similarity(vectors[1], vectors[2])
    
    print(f"'machine learning' vs 'artificial intelligence': {sim_1_2:.4f}")
    print(f"'machine learning' vs 'pizza recipe': {sim_1_3:.4f}")
    print(f"'artificial intelligence' vs 'pizza recipe': {sim_2_3:.4f}")
    
    print("\n💡 Notice: Similar topics have higher scores!")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("VECTOR STORE RETRIEVER - COMPLETE DEMO")
    print("🚀"*40)
    
    # Check if OpenAI API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("\n✅ OpenAI API key found - running OpenAI demo")
        vector_retriever_openai()
    else:
        print("\n⚠️  No OpenAI API key found - skipping OpenAI demo")
        print("   Set OPENAI_API_KEY in .env to use OpenAI embeddings")
    
    # Always run HuggingFace demo (free, local)
    print("\n✅ Running HuggingFace demo (free, local)")
    vector_retriever_huggingface()
    
    # Show how embeddings work
    demonstrate_embeddings()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Vector stores enable semantic search (meaning-based, not keyword-based)
2. Different search types:
   - similarity: Return most similar docs
   - similarity_score_threshold: Only return docs above threshold
   - mmr: Balance relevance and diversity
3. Key parameters:
   - k: Number of results to return
   - fetch_k: Initial candidates for MMR
   - lambda_mult: Diversity vs relevance tradeoff
4. HuggingFace embeddings are free and run locally
5. OpenAI embeddings are paid but very high quality
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")



