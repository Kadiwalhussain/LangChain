"""
Multi-Query Retriever - Intermediate Example
============================================

Multi-Query Retriever generates multiple variations of the user's query
and retrieves documents for each, then combines the results.

Benefits:
- Overcomes query phrasing limitations
- Retrieves more diverse perspectives
- Increases recall (finds more relevant docs)
- Robust to ambiguous queries

Use Case: Complex queries, when phrasing matters
"""

from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
import logging

# Enable logging to see generated queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# ============================================================================
# Basic Multi-Query Retriever
# ============================================================================

def basic_multi_query():
    """Demonstrate basic multi-query retrieval."""
    
    print("="*80)
    print("MULTI-QUERY RETRIEVER")
    print("="*80)
    
    # Create documents
    documents = [
        Document(
            page_content="LangChain is a framework for building applications with large language models.",
            metadata={"source": "doc1"}
        ),
        Document(
            page_content="You can use LangChain to create chatbots, question answering systems, and agents.",
            metadata={"source": "doc2"}
        ),
        Document(
            page_content="Vector databases store embeddings for semantic search capabilities.",
            metadata={"source": "doc3"}
        ),
        Document(
            page_content="RAG systems combine retrieval and generation for better LLM responses.",
            metadata={"source": "doc4"}
        ),
        Document(
            page_content="Python is the primary language for LangChain development.",
            metadata={"source": "doc5"}
        ),
    ]
    
    print(f"\n📚 Created knowledge base with {len(documents)} documents\n")
    
    # Setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, collection_name="multiquery_demo")
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Try to use Ollama
    try:
        llm = OllamaLLM(model="mistral", temperature=0)
        print("✅ Using Ollama (local LLM)\n")
        
        # Create multi-query retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        
        query = "How can I build chatbots?"
        
        print("="*80)
        print(f"USER QUERY: {query}")
        print("="*80)
        
        print("\n📝 LLM will generate multiple query variations...")
        print("   (Check the logs above to see generated queries)\n")
        
        # Retrieve
        results = multi_query_retriever.get_relevant_documents(query)
        
        print("\n📄 RETRIEVED DOCUMENTS:")
        seen_sources = set()
        for i, doc in enumerate(results, 1):
            if doc.metadata['source'] not in seen_sources:
                print(f"\n{i}. {doc.page_content}")
                print(f"   Source: {doc.metadata['source']}")
                seen_sources.add(doc.metadata['source'])
        
        print("\n💡 Multi-query retrieved documents from multiple query perspectives")
        
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("   Install Ollama to run this demo: https://ollama.ai")
    
    vectorstore.delete_collection()


# ============================================================================
# How Multi-Query Works
# ============================================================================

def how_it_works():
    """Explain the multi-query process."""
    
    print("\n\n" + "="*80)
    print("HOW MULTI-QUERY RETRIEVER WORKS")
    print("="*80)
    
    print("""
STEP-BY-STEP PROCESS:

1. User provides original query
   Example: "What are benefits of exercise?"

2. LLM generates multiple query variations
   Generated queries might be:
   - "What are the advantages of physical activity?"
   - "How does exercise help your health?"
   - "What positive effects does working out have?"
   - "Why should someone exercise regularly?"

3. Each query is sent to the base retriever
   Query 1 → Retriever → Docs A, B, C
   Query 2 → Retriever → Docs B, D, E
   Query 3 → Retriever → Docs C, F, G
   Query 4 → Retriever → Docs D, H, I

4. Results are combined and deduplicated
   Final: Docs A, B, C, D, E, F, G, H, I

5. Return unique documents to user

BENEFITS:

✅ Overcomes Query Phrasing Limitations
   - User says "ML" → Generated: "machine learning"
   - User says "car" → Generated: "automobile", "vehicle"

✅ Retrieves Diverse Perspectives
   - Different angles on the same topic
   - More comprehensive coverage

✅ Increases Recall
   - Finds more relevant documents
   - Reduces chance of missing important info

✅ Handles Ambiguous Queries
   - "Python" → could mean language or snake
   - Multiple queries clarify intent

TRADE-OFFS:

❌ More LLM Calls
   - Generates queries: 1 LLM call
   - More expensive than single query

❌ Slower
   - Multiple retrieval operations
   - Query generation takes time

❌ May Retrieve Too Much
   - More results to process
   - May include less relevant docs
    """)


# ============================================================================
# Custom Query Generation
# ============================================================================

def custom_query_generation():
    """Show how to customize query generation."""
    
    print("\n\n" + "="*80)
    print("CUSTOM QUERY GENERATION")
    print("="*80)
    
    print("""
You can customize how queries are generated using a custom prompt:

from langchain.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template='''You are an AI language model assistant. Your task is to generate 3
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}'''
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=QUERY_PROMPT
)

CUSTOMIZATION OPTIONS:

1. Number of Queries
   - Default: 3-5 queries
   - Adjust based on needs
   - More queries = more coverage but slower

2. Query Style
   - Formal vs casual
   - Technical vs simple
   - Question vs statement

3. Domain-Specific
   - Medical queries → medical terminology
   - Legal queries → legal phrasing
   - Code queries → technical terms

EXAMPLE CUSTOM PROMPTS:

For Technical Documentation:
'''Generate 3 technical variations of: {question}
Focus on specific technical terms and API names.'''

For Customer Support:
'''Generate 3 customer-friendly versions of: {question}
Use simple language and common phrases.'''

For Research:
'''Generate 3 academic reformulations of: {question}
Use formal language and research terminology.'''
    """)


# ============================================================================
# Comparison: Single vs Multi-Query
# ============================================================================

def compare_single_vs_multi():
    """Compare single query vs multi-query retrieval."""
    
    print("\n\n" + "="*80)
    print("SINGLE QUERY vs MULTI-QUERY COMPARISON")
    print("="*80)
    
    print("""
SCENARIO: User asks "What's ML?"

SINGLE QUERY RETRIEVAL:
┌─────────────────────────┐
│ Query: "What's ML?"     │
└────────────┬────────────┘
             │
             ▼
      [Vector Database]
             │
             ▼
   ┌─────────────────────┐
   │ Top 3 Results:      │
   │ 1. ML basics        │
   │ 2. Machine learning │
   │ 3. Algorithms       │
   └─────────────────────┘

Problems:
❌ "ML" may not match "machine learning" well
❌ Misses documents that use "artificial intelligence"
❌ Limited to one search perspective

MULTI-QUERY RETRIEVAL:
┌───────────────────────────────┐
│ Original: "What's ML?"        │
└──────────────┬────────────────┘
               │
               ▼
        [LLM Query Generator]
               │
      ┌────────┼────────┐
      │        │        │
      ▼        ▼        ▼
  "What is    "Explain   "Machine
   machine     machine    learning
   learning"   learning"  basics"
      │        │        │
      └────────┼────────┘
               ▼
       [Vector Database]
               │
               ▼
      ┌────────────────────┐
      │ Combined Results:  │
      │ 1. ML basics       │
      │ 2. Machine learning│
      │ 3. AI fundamentals │
      │ 4. ML algorithms   │
      │ 5. Deep learning   │
      │ 6. Data science    │
      └────────────────────┘

Benefits:
✅ Better coverage of the topic
✅ Finds semantically related docs
✅ More robust to phrasing

WHEN TO USE EACH:

Use Single Query When:
- Speed is critical
- Query is already well-formed
- Simple, narrow topic
- Budget-conscious (fewer LLM calls)

Use Multi-Query When:
- Quality over speed
- Complex or ambiguous queries
- Broad topic coverage needed
- Can afford extra LLM calls
    """)


# ============================================================================
# Best Practices
# ============================================================================

def multi_query_best_practices():
    """Best practices for multi-query retrieval."""
    
    print("\n\n" + "="*80)
    print("MULTI-QUERY BEST PRACTICES")
    print("="*80)
    
    print("""
1. WHEN TO USE MULTI-QUERY

   Good Use Cases:
   ✅ Complex user questions
   ✅ Ambiguous queries
   ✅ When phrasing matters
   ✅ Broad topic exploration
   ✅ Research and analysis

   Skip Multi-Query When:
   ❌ Simple, direct questions
   ❌ Speed is critical
   ❌ Tight budget (LLM costs)
   ❌ Query is already precise

2. CONFIGURATION TIPS

   Number of Generated Queries:
   - 3-5 queries: Good balance (default)
   - 2-3 queries: Faster, less coverage
   - 5-7 queries: More thorough, slower

   Base Retriever k Parameter:
   - Set k=2-3 for each query
   - Total results = k × num_queries
   - Deduplication reduces final count

3. PROMPT ENGINEERING

   Good Query Generation Prompts:
   ✅ Clear instructions
   ✅ Specify number of queries
   ✅ Define query style
   ✅ Domain-specific guidance

   Example:
   "Generate 3 different phrasings of the question.
    Use both technical and simple language.
    Focus on: {question}"

4. MONITORING AND DEBUGGING

   Enable logging to see generated queries:
   ```python
   import logging
   logging.basicConfig()
   logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
   ```

   Check:
   - Quality of generated queries
   - Diversity of results
   - Retrieval latency
   - Number of unique documents

5. OPTIMIZATION

   Speed Optimization:
   - Cache query generations
   - Run retrievals in parallel (built-in)
   - Limit base retriever k value

   Quality Optimization:
   - Tune query generation prompt
   - Experiment with different LLMs
   - Adjust number of queries
   - Test with real user queries

6. COMMON PATTERNS

   Pattern 1: Multi-Query + Ensemble
   ```python
   # Combine multi-query with ensemble for best results
   ensemble = EnsembleRetriever([vector_ret, bm25_ret])
   multi_query = MultiQueryRetriever.from_llm(
       retriever=ensemble,
       llm=llm
   )
   ```

   Pattern 2: Multi-Query + Reranking
   ```python
   # Generate many candidates, rerank to top k
   multi_query = MultiQueryRetriever.from_llm(
       retriever=base_retriever,
       llm=llm
   )
   reranked = reranker.compress_documents(
       multi_query.get_relevant_documents(query),
       query
   )
   ```

7. COST CONSIDERATIONS

   Costs:
   - 1 LLM call to generate queries
   - Multiple retrieval operations (usually free)
   - Consider using cheaper LLM for query generation

   Cost Reduction:
   - Use smaller model for query generation
   - Cache query generations
   - Only use for complex queries
    """)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("MULTI-QUERY RETRIEVER - COMPLETE DEMO")
    print("🚀"*40)
    
    # Run demos
    basic_multi_query()
    how_it_works()
    custom_query_generation()
    compare_single_vs_multi()
    multi_query_best_practices()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Multi-Query generates multiple query variations automatically

2. Increases recall by finding more relevant documents

3. Overcomes limitations of single query phrasing

4. Trade-off: Better quality but slower and more expensive

5. Best for:
   - Complex queries
   - When phrasing matters
   - Need comprehensive coverage

6. Combine with ensemble or reranking for best results

7. Monitor generated queries to ensure quality
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")

