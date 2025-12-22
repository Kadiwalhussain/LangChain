"""
Simple RAG Chain - Basic Example
=================================

RAG = Retrieval Augmented Generation

This example demonstrates a complete RAG pipeline:
1. Load documents
2. Create embeddings and vector store
3. Set up retriever
4. Combine with LLM for question answering

RAG Flow:
Query → Retriever → Relevant Docs → LLM (with context) → Answer
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Build a Simple RAG System
# ============================================================================

def simple_rag_demo():
    """Complete RAG system demonstration."""
    
    print("="*80)
    print("SIMPLE RAG (RETRIEVAL AUGMENTED GENERATION) SYSTEM")
    print("="*80)
    
    # Step 1: Create Knowledge Base
    print("\n📚 STEP 1: Creating Knowledge Base")
    print("-"*80)
    
    documents = [
        Document(
            page_content="""
            LangChain is a framework for developing applications powered by language models.
            It provides tools for document loading, text splitting, embeddings, vector stores,
            retrievers, and chains. LangChain makes it easy to build RAG systems.
            """,
            metadata={"source": "langchain_intro", "topic": "langchain"}
        ),
        Document(
            page_content="""
            Python is a high-level, interpreted programming language known for its simplicity
            and readability. It's widely used in web development, data science, machine learning,
            and automation. Python was created by Guido van Rossum in 1991.
            """,
            metadata={"source": "python_intro", "topic": "programming"}
        ),
        Document(
            page_content="""
            Machine Learning is a subset of AI that enables systems to learn and improve from
            experience without being explicitly programmed. It uses algorithms to parse data,
            learn from it, and make predictions or decisions. Common types include supervised,
            unsupervised, and reinforcement learning.
            """,
            metadata={"source": "ml_basics", "topic": "ai"}
        ),
        Document(
            page_content="""
            Vector databases store data as high-dimensional vectors, enabling semantic search.
            Unlike traditional databases that match exact keywords, vector databases find
            similar items based on meaning. Popular vector databases include Pinecone, Weaviate,
            Chroma, and Qdrant.
            """,
            metadata={"source": "vector_db_intro", "topic": "databases"}
        ),
        Document(
            page_content="""
            RAG (Retrieval Augmented Generation) combines information retrieval with text generation.
            It retrieves relevant documents from a knowledge base and provides them as context to
            an LLM, which then generates an informed answer. This reduces hallucinations and provides
            up-to-date information.
            """,
            metadata={"source": "rag_explanation", "topic": "ai"}
        )
    ]
    
    print(f"✅ Created knowledge base with {len(documents)} documents")
    for doc in documents:
        print(f"   - {doc.metadata['topic']}: {doc.page_content[:60].strip()}...")
    
    # Step 2: Create Embeddings and Vector Store
    print("\n🔢 STEP 2: Creating Embeddings and Vector Store")
    print("-"*80)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Loaded embedding model: all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="simple_rag_demo"
    )
    print("✅ Created vector store and embedded all documents")
    
    # Step 3: Create Retriever
    print("\n🔍 STEP 3: Setting Up Retriever")
    print("-"*80)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Retrieve top 2 most relevant documents
    )
    print("✅ Created retriever (will fetch top 2 relevant documents)")
    
    # Step 4: Set Up LLM
    print("\n🤖 STEP 4: Setting Up Language Model")
    print("-"*80)
    
    # Try to use Ollama (local LLM)
    try:
        llm = OllamaLLM(model="mistral", temperature=0.2)
        print("✅ Using Ollama with Mistral model")
        print("   (Make sure Ollama is running: ollama serve)")
    except:
        print("⚠️  Ollama not available - demo will show structure only")
        llm = None
    
    if llm is None:
        print("\n❌ Cannot proceed without LLM. Please install Ollama:")
        print("   https://ollama.ai")
        return
    
    # Step 5: Create RAG Chain
    print("\n⛓️  STEP 5: Creating RAG Chain")
    print("-"*80)
    
    # Custom prompt template
    template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means stuff all docs into the prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Return which docs were used
    )
    
    print("✅ Created RAG chain with custom prompt")
    
    # Step 6: Test the RAG System
    print("\n" + "="*80)
    print("TESTING RAG SYSTEM")
    print("="*80)
    
    test_questions = [
        "What is LangChain?",
        "Explain machine learning in simple terms",
        "What is RAG and why is it useful?",
        "What programming language is mentioned?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─'*80}")
        print(f"QUESTION {i}: {question}")
        print('─'*80)
        
        # Get answer
        result = qa_chain.invoke({"query": question})
        
        # Display retrieved documents
        print("\n📄 RETRIEVED DOCUMENTS:")
        for j, doc in enumerate(result['source_documents'], 1):
            print(f"\n   Document {j} (Source: {doc.metadata['source']}):")
            print(f"   {doc.page_content[:150].strip()}...")
        
        # Display answer
        print(f"\n🤖 ANSWER:")
        print(f"   {result['result']}")
    
    # Clean up
    vectorstore.delete_collection()
    print("\n\n✅ Demo completed and cleaned up")


# ============================================================================
# RAG Without Chain (Manual Process)
# ============================================================================

def manual_rag_process():
    """Show what happens under the hood in RAG."""
    
    print("\n\n" + "="*80)
    print("MANUAL RAG PROCESS (Behind the Scenes)")
    print("="*80)
    
    # Setup
    documents = [
        Document(page_content="Paris is the capital of France. It is known for the Eiffel Tower."),
        Document(page_content="London is the capital of the United Kingdom. It has the Big Ben."),
        Document(page_content="Tokyo is the capital of Japan. It is known for its modern technology."),
    ]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, collection_name="manual_rag")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    question = "What is the capital of France?"
    
    print(f"\n❓ Question: {question}")
    
    # Step 1: Retrieve relevant documents
    print("\n📝 Step 1: Retrieve relevant documents")
    retrieved_docs = retriever.get_relevant_documents(question)
    print(f"   Retrieved: {retrieved_docs[0].page_content}")
    
    # Step 2: Construct prompt with context
    print("\n📝 Step 2: Construct prompt with context")
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
    
    print(f"   Prompt:\n{prompt}")
    
    # Step 3: Send to LLM (simulated)
    print("\n📝 Step 3: Send to LLM")
    print("   (LLM processes prompt and generates answer)")
    
    # Step 4: Return answer
    print("\n📝 Step 4: Return answer")
    print("   LLM Answer: 'The capital of France is Paris.'")
    
    print("\n💡 This is exactly what RetrievalQA chain does automatically!")
    
    vectorstore.delete_collection()


# ============================================================================
# Different Chain Types
# ============================================================================

def different_chain_types():
    """Demonstrate different RAG chain types."""
    
    print("\n\n" + "="*80)
    print("DIFFERENT RAG CHAIN TYPES")
    print("="*80)
    
    print("""
1. STUFF CHAIN (Default)
   - Stuffs all retrieved documents into one prompt
   - Simple and works well for small number of docs
   - Limited by context window size
   
   Flow: Docs → All in one prompt → LLM → Answer

2. MAP-REDUCE CHAIN
   - Maps over each document separately
   - Reduces results into final answer
   - Good for many documents
   
   Flow: Doc1 → LLM → Result1
         Doc2 → LLM → Result2  } → Combine → Final Answer
         Doc3 → LLM → Result3

3. REFINE CHAIN
   - Iteratively refines answer with each doc
   - Builds up answer progressively
   
   Flow: Doc1 → LLM → Draft1
         Draft1 + Doc2 → LLM → Draft2
         Draft2 + Doc3 → LLM → Final Answer

4. MAP-RERANK CHAIN
   - Maps over docs and scores each answer
   - Returns highest scored answer
   
   Flow: Doc1 → LLM → (Answer1, Score1)
         Doc2 → LLM → (Answer2, Score2)  } → Best Answer
         Doc3 → LLM → (Answer3, Score3)
    """)


# ============================================================================
# Best Practices
# ============================================================================

def rag_best_practices():
    """Show RAG best practices."""
    
    print("\n\n" + "="*80)
    print("RAG BEST PRACTICES")
    print("="*80)
    
    print("""
✅ DO:

1. Chunk documents appropriately
   - 500-1500 characters per chunk
   - Add overlap (10-20%) between chunks
   
2. Use good prompts
   - Tell LLM to use context
   - Tell LLM to say "I don't know" if uncertain
   - Keep prompts clear and specific

3. Return source documents
   - Track which docs were used
   - Provides transparency
   - Helps with debugging

4. Set appropriate k value
   - Start with k=3-5
   - More isn't always better
   - Consider context window limits

5. Handle errors gracefully
   - What if no relevant docs found?
   - What if LLM fails?
   - Always have fallback

❌ DON'T:

1. Don't stuff too many docs into context
   - LLM performance degrades with too much context
   - Use map-reduce for many docs

2. Don't use RAG for simple queries
   - "What is 2+2?" doesn't need retrieval
   - Only retrieve when necessary

3. Don't ignore metadata
   - Source tracking is important
   - Metadata helps with filtering

4. Don't use low-quality embeddings
   - Good embeddings = good retrieval
   - Bad embeddings = irrelevant context

5. Don't forget to test
   - Test with various questions
   - Check retrieved documents
   - Validate answer quality
    """)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("SIMPLE RAG SYSTEM - COMPLETE DEMO")
    print("🚀"*40)
    
    # Main RAG demo
    simple_rag_demo()
    
    # Show manual process
    manual_rag_process()
    
    # Show chain types
    different_chain_types()
    
    # Show best practices
    rag_best_practices()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. RAG = Retrieval + Generation
   - Retrieves relevant documents
   - Provides them as context to LLM
   - LLM generates informed answer

2. RAG Benefits:
   ✅ Reduces hallucinations
   ✅ Provides up-to-date information
   ✅ Sources can be traced
   ✅ Domain-specific knowledge

3. Basic RAG Pipeline:
   Documents → Embeddings → Vector Store → Retriever → LLM → Answer

4. Key Components:
   - Knowledge base (documents)
   - Embedding model
   - Vector store
   - Retriever
   - LLM
   - Prompt template

5. Chain Types:
   - Stuff: Simple, all docs in one prompt
   - Map-Reduce: Multiple LLM calls, good for many docs
   - Refine: Iterative refinement
   - Map-Rerank: Score and pick best answer
    """)
    
    print("\n" + "🎉"*40)
    print("Demo completed successfully!")
    print("🎉"*40 + "\n")

