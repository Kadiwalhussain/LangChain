"""
Custom Retriever - Advanced Example
===================================

Create your own custom retriever with any logic you need.

Use Cases:
- API-based retrieval
- Database queries
- Custom scoring algorithms
- Business logic integration
- Hybrid approaches with custom rules
"""

from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
import time

# ============================================================================
# Basic Custom Retriever
# ============================================================================

class SimpleCustomRetriever(BaseRetriever):
    """A simple custom retriever that searches a hardcoded list."""
    
    documents: List[Document]
    k: int = 3
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents relevant to the query."""
        # Simple keyword matching
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            # Check if any query word is in the document
            if any(word in doc.page_content.lower() for word in query_lower.split()):
                results.append(doc)
        
        # Return top k
        return results[:self.k]


def test_simple_custom():
    """Test the simple custom retriever."""
    
    print("="*80)
    print("SIMPLE CUSTOM RETRIEVER")
    print("="*80)
    
    # Create documents
    docs = [
        Document(page_content="Python is a programming language", metadata={"id": 1}),
        Document(page_content="Java is used for enterprise applications", metadata={"id": 2}),
        Document(page_content="Machine learning with Python", metadata={"id": 3}),
    ]
    
    # Create retriever
    retriever = SimpleCustomRetriever(documents=docs, k=2)
    
    query = "Python programming"
    print(f"\n🔍 Query: {query}")
    
    results = retriever.get_relevant_documents(query)
    
    print(f"\n📄 Retrieved {len(results)} documents:")
    for doc in results:
        print(f"   - {doc.page_content}")


# ============================================================================
# Advanced Custom Retriever with Scoring
# ============================================================================

class ScoredCustomRetriever(BaseRetriever):
    """Custom retriever with custom scoring logic."""
    
    documents: List[Document]
    k: int = 3
    boost_recent: bool = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve and rank documents with custom logic."""
        
        # Score each document
        scored_docs = []
        query_words = set(query.lower().split())
        
        for doc in self.documents:
            score = self._calculate_score(doc, query_words)
            scored_docs.append((doc, score))
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, score in scored_docs[:self.k]]
    
    def _calculate_score(self, doc: Document, query_words: set) -> float:
        """Calculate custom relevance score."""
        score = 0.0
        content_lower = doc.page_content.lower()
        
        # 1. Keyword matching (base score)
        word_matches = sum(1 for word in query_words if word in content_lower)
        score += word_matches * 10
        
        # 2. Bonus for title matches (if in metadata)
        if "title" in doc.metadata:
            title_lower = doc.metadata["title"].lower()
            title_matches = sum(1 for word in query_words if word in title_lower)
            score += title_matches * 20  # Title matches worth more
        
        # 3. Recency bonus (if boost_recent is True)
        if self.boost_recent and "year" in doc.metadata:
            year = doc.metadata["year"]
            # Boost recent documents
            recency_score = (year - 2020) * 5  # 5 points per year after 2020
            score += max(0, recency_score)
        
        # 4. Content length penalty (prefer concise answers)
        length = len(doc.page_content)
        if length > 200:
            score -= (length - 200) * 0.01  # Small penalty for long docs
        
        return score


def test_scored_custom():
    """Test the scored custom retriever."""
    
    print("\n\n" + "="*80)
    print("SCORED CUSTOM RETRIEVER")
    print("="*80)
    
    docs = [
        Document(
            page_content="Python tutorial for beginners",
            metadata={"title": "Python Basics", "year": 2024, "id": 1}
        ),
        Document(
            page_content="Advanced Python programming with detailed examples and comprehensive coverage",
            metadata={"title": "Advanced Python", "year": 2022, "id": 2}
        ),
        Document(
            page_content="Python is great",
            metadata={"title": "Python Review", "year": 2023, "id": 3}
        ),
        Document(
            page_content="Java programming tutorial",
            metadata={"title": "Java Guide", "year": 2024, "id": 4}
        ),
    ]
    
    retriever = ScoredCustomRetriever(documents=docs, k=3, boost_recent=True)
    
    query = "Python tutorial"
    print(f"\n🔍 Query: {query}")
    print("\nScoring factors:")
    print("   - Keyword matches: +10 per match")
    print("   - Title matches: +20 per match")
    print("   - Recent docs: +5 per year after 2020")
    print("   - Length penalty: -0.01 per char over 200")
    
    results = retriever.get_relevant_documents(query)
    
    print(f"\n📄 Top {len(results)} documents (ranked by custom score):")
    for i, doc in enumerate(results, 1):
        print(f"\n   {i}. {doc.page_content}")
        print(f"      Title: {doc.metadata['title']}, Year: {doc.metadata['year']}")


# ============================================================================
# API-Based Custom Retriever
# ============================================================================

class MockAPIRetriever(BaseRetriever):
    """Simulates retrieving from an external API."""
    
    api_endpoint: str
    api_key: str
    k: int = 5
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents from API."""
        
        # Simulate API call
        print(f"\n   📡 Calling API: {self.api_endpoint}")
        print(f"   🔑 Using API key: {self.api_key[:10]}...")
        print(f"   ⏳ Fetching results...")
        
        time.sleep(0.5)  # Simulate network latency
        
        # Simulate API response
        mock_results = [
            {
                "text": f"API result 1 for query: {query}",
                "score": 0.95,
                "source": "api_database"
            },
            {
                "text": f"API result 2 related to: {query}",
                "score": 0.87,
                "source": "api_database"
            },
            {
                "text": f"API result 3 about: {query}",
                "score": 0.76,
                "source": "api_database"
            },
        ]
        
        # Convert to LangChain documents
        documents = [
            Document(
                page_content=result["text"],
                metadata={"score": result["score"], "source": result["source"]}
            )
            for result in mock_results[:self.k]
        ]
        
        print(f"   ✅ Retrieved {len(documents)} documents from API\n")
        
        return documents


def test_api_retriever():
    """Test API-based retriever."""
    
    print("\n\n" + "="*80)
    print("API-BASED CUSTOM RETRIEVER")
    print("="*80)
    
    retriever = MockAPIRetriever(
        api_endpoint="https://api.example.com/search",
        api_key="sk_test_1234567890",
        k=3
    )
    
    query = "machine learning algorithms"
    print(f"\n🔍 Query: {query}")
    
    results = retriever.get_relevant_documents(query)
    
    print("📄 Results:")
    for doc in results:
        print(f"   - {doc.page_content}")
        print(f"     Score: {doc.metadata['score']}")


# ============================================================================
# Template for Production Custom Retriever
# ============================================================================

def show_production_template():
    """Show a production-ready custom retriever template."""
    
    print("\n\n" + "="*80)
    print("PRODUCTION CUSTOM RETRIEVER TEMPLATE")
    print("="*80)
    
    template = '''
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionCustomRetriever(BaseRetriever):
    """Production-ready custom retriever template."""
    
    # Configuration
    data_source: Any  # Your data source (DB, API, etc.)
    k: int = 5
    score_threshold: Optional[float] = 0.5
    timeout: int = 30
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Main retrieval logic."""
        try:
            # 1. Preprocess query
            processed_query = self._preprocess_query(query)
            logger.info(f"Processed query: {processed_query}")
            
            # 2. Retrieve candidates
            candidates = self._retrieve_candidates(processed_query)
            logger.info(f"Retrieved {len(candidates)} candidates")
            
            # 3. Score and rank
            scored = self._score_documents(candidates, query)
            
            # 4. Filter by threshold
            if self.score_threshold:
                scored = [
                    (doc, score) for doc, score in scored 
                    if score >= self.score_threshold
                ]
            
            # 5. Return top k
            top_docs = [doc for doc, score in scored[:self.k]]
            logger.info(f"Returning {len(top_docs)} documents")
            
            return top_docs
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []  # Graceful failure
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Add your preprocessing logic
        return query.strip().lower()
    
    def _retrieve_candidates(self, query: str) -> List[Document]:
        """Retrieve candidate documents from your source."""
        # Implement your retrieval logic
        # Could be: database query, API call, file search, etc.
        pass
    
    def _score_documents(
        self, 
        documents: List[Document], 
        query: str
    ) -> List[tuple[Document, float]]:
        """Score and rank documents."""
        # Implement your scoring logic
        # Return list of (document, score) tuples
        pass
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async version of retrieval."""
        # Implement async version if needed
        # Useful for API calls or database queries
        pass


# Usage
retriever = ProductionCustomRetriever(
    data_source=my_database,
    k=5,
    score_threshold=0.7,
    timeout=30
)

results = retriever.get_relevant_documents("query")
'''
    
    print(template)
    
    print("\n" + "="*80)
    print("KEY FEATURES OF PRODUCTION RETRIEVER:")
    print("="*80)
    print("""
✅ Error handling - Graceful failures
✅ Logging - Track what's happening
✅ Timeouts - Don't hang forever
✅ Score threshold - Filter low-quality results
✅ Preprocessing - Clean queries
✅ Async support - For concurrent operations
✅ Configurable - Easy to tune
✅ Type hints - Better IDE support
    """)


# ============================================================================
# Best Practices
# ============================================================================

def custom_retriever_best_practices():
    """Show best practices for custom retrievers."""
    
    print("\n" + "="*80)
    print("CUSTOM RETRIEVER BEST PRACTICES")
    print("="*80)
    
    print("""
1. WHEN TO BUILD CUSTOM RETRIEVER

   Build Custom When:
   ✅ Existing retrievers don't fit your needs
   ✅ Complex business logic required
   ✅ Custom data source (API, special database)
   ✅ Unique scoring requirements
   ✅ Need fine-grained control

   Use Built-in When:
   ❌ Standard vector search suffices
   ❌ Time-constrained
   ❌ Simpler is better

2. DESIGN PRINCIPLES

   Keep It Simple:
   - Start with basic implementation
   - Add complexity only when needed
   - Document your logic

   Make It Testable:
   - Unit test scoring logic
   - Test edge cases
   - Mock external dependencies

   Handle Errors:
   - Catch exceptions
   - Log errors
   - Return empty list on failure (don't crash)

3. PERFORMANCE CONSIDERATIONS

   Speed:
   - Cache frequent queries
   - Use async for I/O operations
   - Limit result set size
   - Set timeouts

   Scalability:
   - Consider batch processing
   - Use connection pooling
   - Implement rate limiting

4. COMMON PATTERNS

   Pattern 1: Hybrid Scoring
   ```python
   score = (
       keyword_score * 0.3 +
       semantic_score * 0.5 +
       recency_score * 0.2
   )
   ```

   Pattern 2: Fallback Chain
   ```python
   results = primary_source()
   if not results:
       results = fallback_source()
   ```

   Pattern 3: Filter Then Rank
   ```python
   candidates = broad_search(query)
   filtered = apply_filters(candidates)
   ranked = score_and_sort(filtered)
   return top_k(ranked)
   ```

5. TESTING

   Unit Tests:
   - Test scoring logic
   - Test edge cases (empty query, no results)
   - Test error handling

   Integration Tests:
   - Test with real data source
   - Test end-to-end retrieval
   - Test performance

   A/B Testing:
   - Compare against baseline
   - Measure user satisfaction
   - Track key metrics

6. MONITORING

   Track:
   - Retrieval latency
   - Error rate
   - Result quality
   - Cache hit rate
   - API costs (if applicable)

7. DOCUMENTATION

   Document:
   - What data source you're using
   - Scoring algorithm details
   - Configuration options
   - Expected behavior
   - Known limitations
    """)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("CUSTOM RETRIEVER - COMPLETE DEMO")
    print("🚀"*40)
    
    # Run demos
    test_simple_custom()
    test_scored_custom()
    test_api_retriever()
    show_production_template()
    custom_retriever_best_practices()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Custom retrievers give you full control over retrieval logic

2. Inherit from BaseRetriever and implement _get_relevant_documents()

3. Use custom retrievers for:
   - Unique data sources
   - Custom scoring algorithms
   - Business logic integration
   - API-based retrieval

4. Follow best practices:
   - Error handling
   - Logging
   - Testing
   - Documentation

5. Start simple, add complexity as needed

6. Monitor performance and quality in production
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")

