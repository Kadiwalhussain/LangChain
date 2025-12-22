"""
Ensemble Retriever - Intermediate Example
=========================================

Ensemble Retriever combines multiple retrievers (e.g., vector + keyword) for better results.

How it works:
1. Multiple retrievers run in parallel
2. Results are combined using Reciprocal Rank Fusion (RRF)
3. Final ranking balances different retrieval methods

Benefits:
- Best of both worlds (semantic + lexical)
- More robust to different query types
- Higher retrieval quality

Use Case: Production RAG systems, hybrid search
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ============================================================================
# Basic Ensemble: Vector + BM25
# ============================================================================

def basic_ensemble():
    """Combine vector and BM25 retrievers."""
    
    print("="*80)
    print("ENSEMBLE RETRIEVER: Vector + BM25")
    print("="*80)
    
    # Sample documents
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on data.",
            metadata={"id": 1, "topic": "AI"}
        ),
        Document(
            page_content="Python is a programming language used in data science and ML.",
            metadata={"id": 2, "topic": "programming"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological brains.",
            metadata={"id": 3, "topic": "AI"}
        ),
        Document(
            page_content="Deep learning uses multiple layers of neural networks for complex tasks.",
            metadata={"id": 4, "topic": "AI"}
        ),
        Document(
            page_content="JavaScript is commonly used for web development and interactive sites.",
            metadata={"id": 5, "topic": "programming"}
        )
    ]
    
    print(f"\n📚 Created {len(documents)} documents\n")
    
    # Create vector retriever (semantic search)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, collection_name="ensemble_demo")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("✅ Created vector retriever (semantic search)")
    
    # Create BM25 retriever (keyword search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    print("✅ Created BM25 retriever (keyword search)")
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Equal weight to both
    )
    
    print("✅ Created ensemble retriever (combines both)")
    
    # Test query
    query = "What is machine learning?"
    
    print(f"\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    # Get results from each retriever individually
    print("\n📊 VECTOR RETRIEVER RESULTS (Semantic):")
    vector_results = vector_retriever.get_relevant_documents(query)
    for i, doc in enumerate(vector_results, 1):
        print(f"   {i}. [ID {doc.metadata['id']}] {doc.page_content[:70]}...")
    
    print("\n📊 BM25 RETRIEVER RESULTS (Keyword):")
    bm25_results = bm25_retriever.get_relevant_documents(query)
    for i, doc in enumerate(bm25_results, 1):
        print(f"   {i}. [ID {doc.metadata['id']}] {doc.page_content[:70]}...")
    
    print("\n📊 ENSEMBLE RESULTS (Combined):")
    ensemble_results = ensemble_retriever.get_relevant_documents(query)
    for i, doc in enumerate(ensemble_results, 1):
        print(f"   {i}. [ID {doc.metadata['id']}] {doc.page_content[:70]}...")
    
    print("\n💡 Notice: Ensemble combines and reranks results from both retrievers")
    
    # Clean up
    vectorstore.delete_collection()


# ============================================================================
# Understanding Reciprocal Rank Fusion (RRF)
# ============================================================================

def demonstrate_rrf():
    """Explain how RRF combines rankings."""
    
    print("\n\n" + "="*80)
    print("RECIPROCAL RANK FUSION (RRF) ALGORITHM")
    print("="*80)
    
    print("""
RRF combines rankings from multiple retrievers using this formula:

    RRF_score(doc) = Σ (1 / (k + rank_i))
    
Where:
- k is a constant (typically 60)
- rank_i is the rank of the document in retriever i
- Σ sums across all retrievers

Example:

Document A:
  - Ranked #1 in Vector search: 1/(60+1) = 0.0164
  - Ranked #3 in BM25: 1/(60+3) = 0.0159
  - Total RRF score: 0.0323

Document B:
  - Ranked #2 in Vector search: 1/(60+2) = 0.0161
  - Ranked #2 in BM25: 1/(60+2) = 0.0161
  - Total RRF score: 0.0322

Result: Document A ranked higher (more consistent across both methods)

Key Insight:
- Documents that rank well in multiple retrievers get higher scores
- This balances different search strategies
- More robust than single retriever
    """)


# ============================================================================
# Tuning Ensemble Weights
# ============================================================================

def tune_ensemble_weights():
    """Show how different weights affect results."""
    
    print("\n\n" + "="*80)
    print("TUNING ENSEMBLE WEIGHTS")
    print("="*80)
    
    documents = [
        Document(page_content="Python programming language", metadata={"id": 1}),
        Document(page_content="Coding with Python for beginners", metadata={"id": 2}),
        Document(page_content="Software development using Python", metadata={"id": 3}),
        Document(page_content="Python snake is a reptile", metadata={"id": 4}),
    ]
    
    # Setup retrievers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, collection_name="weights_demo")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4
    
    query = "Python programming"
    
    # Test different weight configurations
    weight_configs = [
        ([1.0, 0.0], "100% Vector, 0% BM25"),
        ([0.7, 0.3], "70% Vector, 30% BM25"),
        ([0.5, 0.5], "50% Vector, 50% BM25 (Balanced)"),
        ([0.3, 0.7], "30% Vector, 70% BM25"),
        ([0.0, 1.0], "0% Vector, 100% BM25"),
    ]
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📚 Testing {len(weight_configs)} weight configurations:\n")
    
    for weights, description in weight_configs:
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )
        
        results = ensemble.get_relevant_documents(query)
        
        print(f"\n{description}:")
        for i, doc in enumerate(results[:2], 1):  # Show top 2
            print(f"   {i}. [ID {doc.metadata['id']}] {doc.page_content}")
    
    print("\n💡 Recommendations:")
    print("   - Start with [0.5, 0.5] (balanced)")
    print("   - If semantic understanding is key: [0.7, 0.3]")
    print("   - If exact keywords matter: [0.3, 0.7]")
    print("   - Test and tune based on your data")
    
    vectorstore.delete_collection()


# ============================================================================
# Three-Retriever Ensemble
# ============================================================================

def multi_retriever_ensemble():
    """Combine three or more retrievers."""
    
    print("\n\n" + "="*80)
    print("MULTI-RETRIEVER ENSEMBLE (3+ Retrievers)")
    print("="*80)
    
    documents = [
        Document(page_content="Recent AI breakthroughs in 2024", metadata={"year": 2024, "id": 1}),
        Document(page_content="Historical AI developments from 2020", metadata={"year": 2020, "id": 2}),
        Document(page_content="AI trends and predictions for 2024", metadata={"year": 2024, "id": 3}),
        Document(page_content="Machine learning basics explained", metadata={"year": 2022, "id": 4}),
    ]
    
    # Retriever 1: Vector search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, collection_name="multi_demo")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Retriever 2: BM25 keyword search
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    # Retriever 3: Simulated time-weighted (prioritize recent)
    # In practice, you'd use TimeWeightedVectorStoreRetriever
    def recent_first_retriever(docs, query):
        """Simple retriever that prioritizes recent documents."""
        scored = [(doc, doc.metadata.get('year', 0)) for doc in docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]
    
    # For demo, we'll use BM25 with different k as third retriever
    bm25_retriever_alt = BM25Retriever.from_documents(documents)
    bm25_retriever_alt.k = 2
    
    # Create ensemble with 3 retrievers
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever, bm25_retriever_alt],
        weights=[0.4, 0.4, 0.2]  # More weight on first two
    )
    
    query = "AI developments in 2024"
    
    print(f"\n🔍 Query: '{query}'")
    print("\nUsing 3 retrievers:")
    print("   1. Vector search (40%)")
    print("   2. BM25 keyword (40%)")  
    print("   3. Alternative BM25 (20%)")
    
    results = ensemble.get_relevant_documents(query)
    
    print(f"\n📊 Combined results:")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. [Year {doc.metadata['year']}] {doc.page_content}")
    
    vectorstore.delete_collection()


# ============================================================================
# Ensemble Best Practices
# ============================================================================

def ensemble_best_practices():
    """Show best practices for ensemble retrievers."""
    
    print("\n\n" + "="*80)
    print("ENSEMBLE RETRIEVER BEST PRACTICES")
    print("="*80)
    
    print("""
1. CHOOSING RETRIEVERS TO COMBINE

   Good Combinations:
   ✅ Vector + BM25 (most common)
   ✅ Multiple vector stores with different embeddings
   ✅ Vector + Metadata filter + Keyword
   ✅ General retriever + Domain-specific retriever
   
   Less Useful:
   ❌ Multiple identical retrievers
   ❌ Retrievers that are too similar

2. SETTING WEIGHTS

   Equal Weights [0.5, 0.5]:
   - Start here
   - Good default for balanced search
   
   Vector-Heavy [0.7, 0.3]:
   - When semantic understanding is critical
   - Natural language queries
   - Synonym/paraphrase matching important
   
   Keyword-Heavy [0.3, 0.7]:
   - Technical terms, IDs, codes
   - Exact phrase matching important
   - Domain-specific terminology

3. PERFORMANCE CONSIDERATIONS

   Speed:
   - Ensemble runs retrievers in parallel
   - Speed ≈ slowest retriever
   - Consider caching

   Accuracy:
   - Usually 10-30% better than single retriever
   - Worth the added complexity for production

4. TESTING & TUNING

   Steps:
   1. Test each retriever individually
   2. Start with equal weights
   3. Collect real user queries
   4. Evaluate retrieval quality
   5. Adjust weights based on results
   6. A/B test in production

5. WHEN TO USE ENSEMBLE

   Use Ensemble When:
   ✅ Quality is more important than simplicity
   ✅ You have diverse query types
   ✅ Single retriever isn't good enough
   ✅ Production RAG system
   
   Skip Ensemble When:
   ✅ Simple use case
   ✅ Single retriever works well
   ✅ Speed is critical
   ✅ Resource-constrained environment

6. COMMON PATTERNS

   Pattern 1: Semantic + Lexical
   ```python
   ensemble = EnsembleRetriever(
       retrievers=[vector_retriever, bm25_retriever],
       weights=[0.6, 0.4]
   )
   ```
   
   Pattern 2: Multiple Embeddings
   ```python
   ensemble = EnsembleRetriever(
       retrievers=[openai_retriever, huggingface_retriever],
       weights=[0.5, 0.5]
   )
   ```
   
   Pattern 3: General + Specific
   ```python
   ensemble = EnsembleRetriever(
       retrievers=[general_retriever, domain_retriever],
       weights=[0.4, 0.6]
   )
   ```

7. MONITORING

   Track:
   - Retrieval latency
   - Result overlap between retrievers
   - User satisfaction/clicks
   - Query types that work well/poorly
    """)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("ENSEMBLE RETRIEVER - COMPLETE DEMO")
    print("🚀"*40)
    
    # Run all demos
    basic_ensemble()
    demonstrate_rrf()
    tune_ensemble_weights()
    multi_retriever_ensemble()
    ensemble_best_practices()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Ensemble combines multiple retrievers for better results

2. Reciprocal Rank Fusion (RRF) merges rankings intelligently

3. Common combination: Vector (semantic) + BM25 (keyword)

4. Start with equal weights [0.5, 0.5], then tune

5. Benefits:
   ✅ More robust to different query types
   ✅ Better recall and precision
   ✅ Combines strengths of different methods

6. Use in production for best RAG quality

7. Weight tuning guidelines:
   - Semantic queries → Higher vector weight
   - Exact matches → Higher keyword weight
   - Balanced → Equal weights
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")

