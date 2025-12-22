"""
Keyword Retriever (BM25) - Basic Example
========================================

This example demonstrates BM25 retriever, a keyword-based (lexical) search algorithm.

How it works:
1. Uses TF-IDF style scoring (Term Frequency - Inverse Document Frequency)
2. Scores documents based on exact keyword matches
3. No semantic understanding (doesn't understand meaning)
4. Fast and effective for exact term matching

When to use:
- Searching for specific keywords or technical terms
- When exact matches are important
- As part of hybrid search (combined with vector search)

Comparison:
- Vector Search: Understands meaning ("ML" matches "machine learning")
- BM25: Only matches exact words ("ML" doesn't match "machine learning")
"""

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

# ============================================================================
# Basic BM25 Retriever
# ============================================================================

def basic_bm25():
    """Demonstrate basic BM25 retriever functionality."""
    
    print("="*80)
    print("BASIC BM25 KEYWORD RETRIEVER")
    print("="*80)
    
    # Create sample documents
    documents = [
        Document(
            page_content="Python is a high-level programming language. Python is easy to learn.",
            metadata={"title": "Python Intro", "category": "programming"}
        ),
        Document(
            page_content="JavaScript is used for web development and creating interactive websites.",
            metadata={"title": "JavaScript Basics", "category": "programming"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence focused on data.",
            metadata={"title": "ML Overview", "category": "AI"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers for complex tasks.",
            metadata={"title": "Deep Learning", "category": "AI"}
        ),
        Document(
            page_content="The Python programming language is widely used in data science and machine learning.",
            metadata={"title": "Python in Data Science", "category": "programming"}
        )
    ]
    
    print(f"\n📚 Created {len(documents)} documents")
    
    # Create BM25 retriever
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 3  # Return top 3 results
    
    print("✅ BM25 Retriever created")
    
    # Test Query 1: Exact keyword match
    print("\n" + "="*80)
    print("TEST 1: Exact Keyword Match")
    print("="*80)
    
    query1 = "Python programming"
    print(f"\n🔍 Query: '{query1}'")
    
    results1 = retriever.get_relevant_documents(query1)
    
    print(f"\n📄 Retrieved {len(results1)} documents:\n")
    for i, doc in enumerate(results1, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Title: {doc.metadata['title']}")
        print(f"   Category: {doc.metadata['category']}\n")
    
    print("💡 Notice: Documents with 'Python' and 'programming' ranked highest")
    
    # Test Query 2: Single keyword
    print("="*80)
    print("TEST 2: Single Keyword")
    print("="*80)
    
    query2 = "learning"
    print(f"\n🔍 Query: '{query2}'")
    
    results2 = retriever.get_relevant_documents(query2)
    
    print(f"\n📄 Retrieved {len(results2)} documents:\n")
    for i, doc in enumerate(results2, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Matches 'learning': {query2 in doc.page_content.lower()}\n")
    
    # Test Query 3: Multiple keywords
    print("="*80)
    print("TEST 3: Multiple Keywords")
    print("="*80)
    
    query3 = "neural networks deep"
    print(f"\n🔍 Query: '{query3}'")
    
    results3 = retriever.get_relevant_documents(query3)
    
    print(f"\n📄 Retrieved {len(results3)} documents:\n")
    for i, doc in enumerate(results3, 1):
        # Count keyword matches
        matches = sum(word in doc.page_content.lower() for word in query3.split())
        print(f"{i}. {doc.page_content}")
        print(f"   Keyword matches: {matches}/{len(query3.split())}\n")


# ============================================================================
# BM25 vs Vector Search Comparison
# ============================================================================

def compare_bm25_vs_vector():
    """Compare BM25 (keyword) vs Vector (semantic) search."""
    
    print("\n\n" + "="*80)
    print("BM25 vs VECTOR SEARCH COMPARISON")
    print("="*80)
    
    documents = [
        Document(page_content="The car is red and fast", metadata={"id": 1}),
        Document(page_content="The automobile is crimson colored", metadata={"id": 2}),
        Document(page_content="Machine learning models learn from data", metadata={"id": 3}),
    ]
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2
    
    query = "red car"
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📚 Documents:")
    for doc in documents:
        print(f"  {doc.metadata['id']}. {doc.page_content}")
    
    # BM25 Results
    print("\n" + "-"*80)
    print("BM25 RETRIEVER RESULTS (Keyword-based)")
    print("-"*80)
    
    bm25_results = bm25_retriever.get_relevant_documents(query)
    
    print(f"\nTop {len(bm25_results)} results:")
    for i, doc in enumerate(bm25_results, 1):
        print(f"{i}. Doc {doc.metadata['id']}: {doc.page_content}")
    
    print("\n💡 BM25 matches 'red' and 'car' literally")
    print("   It WON'T match Doc 2 even though it's semantically similar")
    print("   (because 'crimson' ≠ 'red' and 'automobile' ≠ 'car' in keyword matching)")
    
    print("\n" + "-"*80)
    print("VECTOR SEARCH WOULD:")
    print("-"*80)
    print("✅ Understand 'crimson' means 'red'")
    print("✅ Understand 'automobile' means 'car'")
    print("✅ Match Doc 2 as highly relevant")
    print("\n💡 This is why hybrid search (BM25 + Vector) works best!")


# ============================================================================
# Advanced BM25 Configuration
# ============================================================================

def advanced_bm25():
    """Demonstrate advanced BM25 configurations."""
    
    print("\n\n" + "="*80)
    print("ADVANCED BM25 CONFIGURATION")
    print("="*80)
    
    # Larger document set
    documents = [
        Document(page_content="Python tutorial for beginners", metadata={"type": "tutorial"}),
        Document(page_content="Advanced Python programming guide", metadata={"type": "guide"}),
        Document(page_content="Python best practices and patterns", metadata={"type": "guide"}),
        Document(page_content="JavaScript fundamentals tutorial", metadata={"type": "tutorial"}),
        Document(page_content="Java programming basics", metadata={"type": "tutorial"}),
        Document(page_content="Python web development with Django", metadata={"type": "guide"}),
        Document(page_content="Data analysis using Python pandas", metadata={"type": "tutorial"}),
        Document(page_content="Machine learning with Python scikit-learn", metadata={"type": "tutorial"}),
    ]
    
    print(f"\n📚 Document collection: {len(documents)} documents")
    
    # Configuration 1: Return top 5
    print("\n" + "-"*80)
    print("CONFIGURATION 1: Return Top 5")
    print("-"*80)
    
    retriever1 = BM25Retriever.from_documents(documents)
    retriever1.k = 5
    
    query = "Python tutorial"
    print(f"\n🔍 Query: '{query}'")
    
    results1 = retriever1.get_relevant_documents(query)
    
    print(f"\nTop 5 results:")
    for i, doc in enumerate(results1, 1):
        print(f"{i}. {doc.page_content} (Type: {doc.metadata['type']})")
    
    # Configuration 2: Filter by metadata
    print("\n" + "-"*80)
    print("CONFIGURATION 2: Combining with Metadata Filtering")
    print("-"*80)
    
    # Get results and filter
    all_results = retriever1.get_relevant_documents(query)
    tutorial_only = [doc for doc in all_results if doc.metadata["type"] == "tutorial"]
    
    print(f"\n🔍 Query: '{query}' (tutorials only)")
    print(f"\nFiltered to {len(tutorial_only)} tutorials:")
    for i, doc in enumerate(tutorial_only, 1):
        print(f"{i}. {doc.page_content}")
    
    # Configuration 3: Custom preprocessing
    print("\n" + "-"*80)
    print("CONFIGURATION 3: Understanding BM25 Scoring")
    print("-"*80)
    
    print("""
BM25 Scoring Factors:
1. Term Frequency (TF): How often does the term appear in the document?
2. Inverse Document Frequency (IDF): How rare is the term across all documents?
3. Document Length: Shorter docs may rank higher for same term frequency

Example:
- Query: "Python tutorial"
- Doc A: "Python" appears 3 times (higher TF)
- Doc B: "Python" appears 1 time (lower TF)
- Doc A will rank higher
    """)


# ============================================================================
# Real-World Use Cases
# ============================================================================

def real_world_examples():
    """Show real-world use cases for BM25."""
    
    print("\n\n" + "="*80)
    print("REAL-WORLD USE CASES")
    print("="*80)
    
    # Use Case 1: FAQ Search
    print("\n" + "-"*80)
    print("USE CASE 1: FAQ Search System")
    print("-"*80)
    
    faq_documents = [
        Document(
            page_content="How do I reset my password? Go to settings and click 'Reset Password'.",
            metadata={"category": "account", "faq_id": 1}
        ),
        Document(
            page_content="How to change email address? Visit profile settings and update your email.",
            metadata={"category": "account", "faq_id": 2}
        ),
        Document(
            page_content="What payment methods are accepted? We accept credit cards, PayPal, and bank transfers.",
            metadata={"category": "billing", "faq_id": 3}
        ),
        Document(
            page_content="How to cancel subscription? Go to billing and click 'Cancel Subscription'.",
            metadata={"category": "billing", "faq_id": 4}
        ),
    ]
    
    faq_retriever = BM25Retriever.from_documents(faq_documents)
    faq_retriever.k = 2
    
    customer_queries = [
        "reset password",
        "payment options",
        "cancel my subscription"
    ]
    
    print("\n📋 FAQ System Demo:")
    for query in customer_queries:
        print(f"\n❓ Customer: {query}")
        results = faq_retriever.get_relevant_documents(query)
        print(f"🤖 Top matching FAQ:")
        print(f"   {results[0].page_content}")
    
    # Use Case 2: Code Search
    print("\n\n" + "-"*80)
    print("USE CASE 2: Code Documentation Search")
    print("-"*80)
    
    code_docs = [
        Document(
            page_content="def calculate_sum(numbers): Returns the sum of a list of numbers",
            metadata={"function": "calculate_sum", "module": "math_utils"}
        ),
        Document(
            page_content="def find_maximum(array): Returns the maximum value in an array",
            metadata={"function": "find_maximum", "module": "array_utils"}
        ),
        Document(
            page_content="def sort_list(items, reverse=False): Sorts a list in ascending or descending order",
            metadata={"function": "sort_list", "module": "list_utils"}
        ),
    ]
    
    code_retriever = BM25Retriever.from_documents(code_docs)
    code_retriever.k = 1
    
    print("\n🔍 Searching code documentation:")
    
    code_query = "function to find maximum value"
    print(f"\nQuery: '{code_query}'")
    code_result = code_retriever.get_relevant_documents(code_query)
    print(f"Result: {code_result[0].page_content}")
    print(f"Function: {code_result[0].metadata['function']}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("BM25 KEYWORD RETRIEVER - COMPLETE DEMO")
    print("🚀"*40)
    
    # Run all demos
    basic_bm25()
    compare_bm25_vs_vector()
    advanced_bm25()
    real_world_examples()
    
    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. BM25 is keyword-based (lexical) search
   - Matches exact words and their stems
   - Fast and deterministic
   - No semantic understanding

2. Best for:
   ✅ Exact term matching (product codes, IDs, names)
   ✅ Technical documentation (where exact terms matter)
   ✅ FAQs (when users search with exact phrases)
   ✅ Hybrid search (combine with vector search)

3. Limitations:
   ❌ Doesn't understand synonyms (car ≠ automobile)
   ❌ Doesn't understand context
   ❌ Sensitive to exact wording

4. Key Parameters:
   - k: Number of results to return
   - Documents are preprocessed (tokenized, lowercased)

5. Production Tip:
   Use BM25 + Vector Search together for best results!
   - BM25: Catches exact matches
   - Vector: Catches semantic matches
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")

