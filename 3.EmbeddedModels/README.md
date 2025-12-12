# Embeddings - Complete Guide

> **Vector Representations**: Convert text to numerical vectors for semantic search and similarity

## 📚 Table of Contents

1. [What are Embeddings?](#what-are-embeddings)
2. [Why Use Embeddings?](#why-use-embeddings)
3. [Folder Structure](#folder-structure)
4. [Supported Providers](#supported-providers)
5. [Core Concepts](#core-concepts)
6. [Examples](#examples)
7. [Use Cases](#use-cases)
8. [Best Practices](#best-practices)

---

## What are Embeddings?

Embeddings are numerical vector representations of text that capture semantic meaning. Similar texts have similar vectors, enabling semantic search and comparison.

```mermaid
graph LR
    A[Text] --> B[Embedding Model]
    B --> C[Vector]
    C --> D[Store/Compare]
    
    style B fill:#FFD700,stroke:#333,stroke-width:2px
```

### Example:
```python
text = "What is machine learning?"
vector = embeddings.embed_query(text)
# Vector: [0.021, -0.034, 0.089, ..., 0.012]  (1536 dimensions)
```

---

## Why Use Embeddings?

| Without Embeddings | With Embeddings |
|-------------------|-----------------|
| Keyword matching only | Semantic understanding |
| "dog" ≠ "puppy" | "dog" ≈ "puppy" |
| Manual similarity rules | Automatic similarity |
| Limited search | Powerful semantic search |

### Key Benefits:
- **Semantic similarity**: Find related content by meaning
- **RAG (Retrieval)**: Power retrieval-augmented generation
- **Clustering**: Group similar documents
- **Classification**: Categorize text automatically

---

## Folder Structure

```
3.EmbeddedModels/
├── openai_embeddings.py           # OpenAI embeddings
├── ollama_embeddings.py           # Ollama local embeddings
├── huggingface_embeddings.py      # HuggingFace models
├── huggingface_instruct_embeddings.py  # Instruction-tuned
├── cohere_embeddings.py           # Cohere embeddings
├── google_embeddings.py           # Google embeddings
├── azure_openai_embeddings.py     # Azure OpenAI
├── vertex_ai_embeddings.py        # Google Vertex AI
├── fastembed_embeddings.py        # FastEmbed (local)
└── README.md                      # This file
```

---

## Supported Providers

### 1. OpenAI (`openai_embeddings.py`)

**Best for**: Production, high-quality embeddings

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single query
query_vector = embeddings.embed_query("What is LangChain?")
print(f"Dimensions: {len(query_vector)}")  # 1536

# Embed multiple documents
docs = ["Python is great", "JavaScript is popular"]
doc_vectors = embeddings.embed_documents(docs)
print(f"Embedded {len(doc_vectors)} documents")
```

**Models:**
| Model | Dimensions | Cost (per 1M tokens) |
|-------|-----------|---------------------|
| text-embedding-3-small | 1536 | $0.02 |
| text-embedding-3-large | 3072 | $0.13 |
| text-embedding-ada-002 | 1536 | $0.10 |

---

### 2. Ollama (`ollama_embeddings.py`)

**Best for**: Local development, free, privacy

```python
from langchain_community.embeddings import OllamaEmbeddings

# Requires Ollama running locally
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed text
vector = embeddings.embed_query("Machine learning basics")
print(f"Dimensions: {len(vector)}")

# Embed multiple texts
texts = ["Python programming", "Data science", "AI research"]
vectors = embeddings.embed_documents(texts)
```

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Start Ollama
ollama serve
```

**Popular Models:**
| Model | Dimensions | Size |
|-------|-----------|------|
| nomic-embed-text | 768 | 274 MB |
| all-minilm | 384 | 45 MB |
| mxbai-embed-large | 1024 | 670 MB |

---

### 3. HuggingFace (`huggingface_embeddings.py`)

**Best for**: Custom models, open-source, fine-tuning

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed text
vector = embeddings.embed_query("What is deep learning?")
print(f"Dimensions: {len(vector)}")  # 384
```

**Popular Models:**
| Model | Dimensions | Performance |
|-------|-----------|-------------|
| all-MiniLM-L6-v2 | 384 | Fast, good quality |
| all-mpnet-base-v2 | 768 | Best quality |
| paraphrase-MiniLM-L6-v2 | 384 | Good for paraphrasing |

---

### 4. Cohere (`cohere_embeddings.py`)

**Best for**: Multilingual, production-ready

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key="your-api-key"
)

vector = embeddings.embed_query("What is NLP?")
print(f"Dimensions: {len(vector)}")  # 1024
```

**Models:**
| Model | Languages | Dimensions |
|-------|-----------|-----------|
| embed-english-v3.0 | English | 1024 |
| embed-multilingual-v3.0 | 100+ | 1024 |

---

### 5. Google (`google_embeddings.py`)

**Best for**: Google ecosystem integration

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vector = embeddings.embed_query("Explain AI")
print(f"Dimensions: {len(vector)}")
```

---

### 6. FastEmbed (`fastembed_embeddings.py`)

**Best for**: Fast local embeddings, CPU-optimized

```python
from langchain_community.embeddings import FastEmbedEmbeddings

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector = embeddings.embed_query("Fast embedding example")
print(f"Dimensions: {len(vector)}")
```

---

## Core Concepts

### 1. Vector Dimensions

Embeddings have fixed dimensions that represent semantic features:

```python
# Different models have different dimensions
openai = OpenAIEmbeddings()  # 1536 dimensions
minilm = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384 dimensions
```

| Dimensions | Trade-off |
|-----------|-----------|
| Lower (384) | Faster, less storage, slightly less accurate |
| Higher (1536+) | More accurate, more storage, slower |

### 2. Semantic Similarity

Similar meanings = Similar vectors:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = OpenAIEmbeddings()

# Similar texts should have high similarity
v1 = embeddings.embed_query("I love dogs")
v2 = embeddings.embed_query("I adore puppies")
v3 = embeddings.embed_query("I like cars")

sim_1_2 = cosine_similarity([v1], [v2])[0][0]  # High (0.9+)
sim_1_3 = cosine_similarity([v1], [v3])[0][0]  # Low (0.3-0.5)

print(f"Dogs vs Puppies: {sim_1_2:.3f}")
print(f"Dogs vs Cars: {sim_1_3:.3f}")
```

### 3. Query vs Document Embeddings

```python
# For searching - embed the query
query_vector = embeddings.embed_query("What is Python?")

# For storing - embed documents
doc_vectors = embeddings.embed_documents([
    "Python is a programming language",
    "JavaScript runs in browsers",
    "Rust is memory-safe"
])
```

### 4. Batch Processing

```python
# Efficient batch embedding
documents = ["doc1", "doc2", "doc3", ..., "doc1000"]
vectors = embeddings.embed_documents(documents)  # One API call
```

---

## Examples

### Basic Embedding

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# Single query
text = "What is machine learning?"
vector = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Vector dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
```

### Similarity Search

```python
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = OpenAIEmbeddings()

# Documents to search
documents = [
    "Python is a programming language",
    "Machine learning uses data to learn patterns",
    "Cats are cute animals",
    "Deep learning is a subset of ML"
]

# Embed all documents
doc_vectors = embeddings.embed_documents(documents)

# Search query
query = "What is artificial intelligence?"
query_vector = embeddings.embed_query(query)

# Calculate similarities
similarities = cosine_similarity([query_vector], doc_vectors)[0]

# Find most similar
for i, (doc, sim) in enumerate(zip(documents, similarities)):
    print(f"{sim:.3f} - {doc}")
```

### With Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

# Create documents
texts = [
    "Python is great for data science",
    "JavaScript powers web development",
    "Rust ensures memory safety",
    "Go is designed for concurrency"
]

# Create vector store
vectorstore = FAISS.from_texts(texts, embeddings)

# Search
results = vectorstore.similarity_search("data analysis language", k=2)
for doc in results:
    print(doc.page_content)
```

### RAG Pattern

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Setup embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Create RAG chain
llm = ChatOpenAI(model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Query
response = qa_chain.invoke("What programming language is best for data?")
print(response)
```

---

## Use Cases

### 1. Semantic Search

```mermaid
graph LR
    A[User Query] --> B[Embed Query]
    B --> C[Compare with Doc Vectors]
    C --> D[Return Similar Docs]
    
    style B fill:#FFD700
```

```python
# Find documents similar to query
query_vector = embeddings.embed_query("machine learning tutorial")
similar_docs = vectorstore.similarity_search(query_vector, k=5)
```

### 2. Document Clustering

```python
from sklearn.cluster import KMeans
import numpy as np

# Embed documents
vectors = embeddings.embed_documents(documents)

# Cluster
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(vectors)

# Group documents by cluster
for i, (doc, cluster) in enumerate(zip(documents, clusters)):
    print(f"Cluster {cluster}: {doc}")
```

### 3. Duplicate Detection

```python
from sklearn.metrics.pairwise import cosine_similarity

vectors = embeddings.embed_documents(documents)
similarity_matrix = cosine_similarity(vectors)

# Find near-duplicates (similarity > 0.95)
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if similarity_matrix[i][j] > 0.95:
            print(f"Possible duplicate: {documents[i]} <-> {documents[j]}")
```

### 4. RAG (Retrieval-Augmented Generation)

```mermaid
graph LR
    A[Question] --> B[Embed]
    B --> C[Find Relevant Docs]
    C --> D[Add to Prompt]
    D --> E[LLM Generates Answer]
    
    style B fill:#FFD700
    style E fill:#87CEEB
```

---

## Best Practices

### 1. Choose the Right Model

| Use Case | Recommended |
|----------|-------------|
| Production, high quality | OpenAI text-embedding-3-small |
| Local development | Ollama nomic-embed-text |
| Cost-sensitive | HuggingFace all-MiniLM-L6-v2 |
| Multilingual | Cohere embed-multilingual-v3.0 |

### 2. Batch Embed When Possible

```python
# ✅ Good - efficient batch call
vectors = embeddings.embed_documents(many_documents)

# ❌ Bad - many API calls
vectors = [embeddings.embed_query(doc) for doc in many_documents]
```

### 3. Cache Embeddings

```python
import hashlib
import json

cache = {}

def get_embedding_cached(text):
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in cache:
        cache[key] = embeddings.embed_query(text)
    return cache[key]
```

### 4. Normalize Vectors

```python
import numpy as np

def normalize(vector):
    return vector / np.linalg.norm(vector)

# Normalized vectors for consistent similarity scores
normalized_vector = normalize(np.array(vector))
```

### 5. Handle Long Texts

```python
# Split long documents before embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(long_document)
vectors = embeddings.embed_documents(chunks)
```

---

## Comparison Table

| Provider | Dimensions | Cost | Speed | Quality |
|----------|-----------|------|-------|---------|
| OpenAI | 1536/3072 | $$ | Fast | ⭐⭐⭐⭐⭐ |
| Ollama | 768+ | Free | Fast | ⭐⭐⭐⭐ |
| HuggingFace | 384-768 | Free | Medium | ⭐⭐⭐⭐ |
| Cohere | 1024 | $$ | Fast | ⭐⭐⭐⭐⭐ |
| FastEmbed | 384-1024 | Free | Fast | ⭐⭐⭐⭐ |

---

## Troubleshooting

### Issue: Dimension mismatch
```python
# Solution: Use same model for query and documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Use this same instance for both query and document embedding
```

### Issue: API rate limiting
```python
# Solution: Batch embeddings and add delays
import time
batch_size = 100
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    vectors = embeddings.embed_documents(batch)
    time.sleep(0.5)  # Rate limit delay
```

### Issue: Out of memory
```python
# Solution: Process in batches
batch_size = 50
all_vectors = []
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    vectors = embeddings.embed_documents(batch)
    all_vectors.extend(vectors)
```

---

## Quick Reference

```python
# OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Ollama (local)
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cohere
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Basic usage
query_vector = embeddings.embed_query("Your query")
doc_vectors = embeddings.embed_documents(["doc1", "doc2"])
```

---

## Next Steps

After mastering Embeddings:
1. **Vector Stores** - Store and search vectors (FAISS, Pinecone, etc.)
2. **RAG** - Build retrieval-augmented generation systems
3. **Prompts** - Combine with prompt templates (see `4.Prompts/`)
4. **Runnables** - Modern composition (see `9.Runnables/`)

---

**Remember:** Embeddings are the foundation of semantic search and RAG. Choose the right model for your use case and always batch your embeddings for efficiency!
