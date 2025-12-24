# Document Loaders - Complete Guide

> **Load data from any source** - Text files, PDFs, Web pages, Databases, APIs, and more

## 📚 Table of Contents

1. [What are Document Loaders?](#what-are-document-loaders)
2. [Architecture Overview](#architecture-overview)
3. [Loader Types](#loader-types)
4. [Basic Loaders](#basic-loaders)
5. [Intermediate Loaders](#intermediate-loaders)
6. [Advanced Loaders](#advanced-loaders)
7. [Best Practices](#best-practices)
8. [Comparison Guide](#comparison-guide)

---

## What are Document Loaders?

Document Loaders are components that load data from various sources and convert it into LangChain `Document` objects.

```mermaid
graph LR
    A[Data Source] --> B[Document Loader]
    B --> C[Document Objects]
    C --> D[Text Splitter]
    D --> E[Vector Store]
    
    style B fill:#FFD700
    style C fill:#90EE90
```

**Document Structure:**
```python
Document(
    page_content="The actual text content",
    metadata={
        "source": "file.pdf",
        "page": 1,
        "author": "John Doe"
    }
)
```

---

## Architecture Overview

### Complete Data Ingestion Pipeline

```mermaid
graph TB
    subgraph "Phase 1: Loading"
        A[Raw Data Source] --> B{Document Loader}
        B -->|File| C[FileLoader]
        B -->|URL| D[WebLoader]
        B -->|DB| E[DatabaseLoader]
        B -->|API| F[APILoader]
    end
    
    subgraph "Phase 2: Processing"
        C --> G[Document Objects]
        D --> G
        E --> G
        F --> G
        G --> H[Add Metadata]
    end
    
    subgraph "Phase 3: Splitting"
        H --> I[Text Splitter]
        I --> J[Document Chunks]
    end
    
    subgraph "Phase 4: Storage"
        J --> K[Embeddings]
        K --> L[Vector Store]
    end
    
    style B fill:#FFD700
    style G fill:#90EE90
    style L fill:#87CEEB
```

---

## Loader Types

### By Data Source

```mermaid
graph TD
    A[Document Loaders] --> B[File-Based]
    A --> C[Web-Based]
    A --> D[Database]
    A --> E[API-Based]
    A --> F[Cloud Storage]
    
    B --> B1[Text, PDF, Word]
    B --> B2[CSV, JSON, XML]
    B --> B3[Markdown, HTML]
    
    C --> C1[Web Pages]
    C --> C2[Sitemap]
    C --> C3[RSS Feeds]
    
    D --> D1[SQL Databases]
    D --> D2[MongoDB]
    D --> D3[ElasticSearch]
    
    E --> E1[REST APIs]
    E --> E2[GraphQL]
    E --> E3[Custom APIs]
    
    F --> F1[S3, GCS]
    F --> F2[Dropbox]
    F --> F3[Google Drive]
    
    style A fill:#FFD700
```

### Comparison Matrix

| Loader Type | Use Case | Complexity | Speed | Metadata |
|-------------|----------|------------|-------|----------|
| **TextLoader** | Simple text files | 🔧 | ⚡⚡⚡ | ⭐ |
| **PDFLoader** | PDF documents | 🔧🔧 | ⚡⚡ | ⭐⭐⭐ |
| **WebLoader** | Web scraping | 🔧🔧 | ⚡⚡ | ⭐⭐ |
| **DirectoryLoader** | Bulk file loading | 🔧 | ⚡⚡⚡ | ⭐⭐ |
| **CSVLoader** | Structured data | 🔧 | ⚡⚡⚡ | ⭐⭐⭐ |
| **JSONLoader** | API responses | 🔧🔧 | ⚡⚡⚡ | ⭐⭐⭐ |
| **SQLDatabaseLoader** | Database queries | 🔧🔧🔧 | ⚡⚡ | ⭐⭐⭐⭐ |
| **NotionLoader** | Notion workspace | 🔧🔧 | ⚡ | ⭐⭐⭐⭐ |

---

## Basic Loaders

### 1. Text Loader

**Load plain text files**

```mermaid
graph LR
    A[Text File] --> B[TextLoader]
    B --> C[Document]
    
    style B fill:#90EE90
```

**Example:**
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

print(documents[0].page_content)
print(documents[0].metadata)
```

**File:** `1.Basic/1_text_loader.py`

---

### 2. PDF Loader

**Extract text from PDFs**

```mermaid
graph TB
    A[PDF File] --> B{PDF Loader}
    B -->|PyPDF| C[Basic Extraction]
    B -->|PyMuPDF| D[Better Quality]
    B -->|PDFPlumber| E[Tables + Text]
    
    C --> F[Documents]
    D --> F
    E --> F
    
    style B fill:#FFD700
    style F fill:#90EE90
```

**Example:**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

for i, page in enumerate(pages):
    print(f"Page {i}: {page.page_content[:100]}...")
    print(f"Metadata: {page.metadata}")
```

**File:** `1.Basic/2_pdf_loader.py`

---

### 3. Directory Loader

**Load multiple files at once**

```mermaid
graph TB
    A[Directory] --> B[DirectoryLoader]
    B --> C[File 1.txt]
    B --> D[File 2.pdf]
    B --> E[File 3.md]
    
    C --> F[Documents List]
    D --> F
    E --> F
    
    style B fill:#FFD700
    style F fill:#90EE90
```

**Example:**
```python
from langchain_community.document_loaders import DirectoryLoader

# Load all text files
loader = DirectoryLoader(
    "documents/",
    glob="**/*.txt",
    show_progress=True
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
```

**File:** `1.Basic/3_directory_loader.py`

---

### 4. CSV Loader

**Load structured CSV data**

```mermaid
graph LR
    A[CSV File] --> B[CSVLoader]
    B --> C[Row 1: Document]
    B --> D[Row 2: Document]
    B --> E[Row N: Document]
    
    style B fill:#FFD700
```

**Example:**
```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["name", "description"]
    }
)

documents = loader.load()
```

**File:** `1.Basic/4_csv_loader.py`

---

## Intermediate Loaders

### 1. Web Loaders

**Scrape content from websites**

```mermaid
graph TB
    A[URL] --> B[WebBaseLoader]
    B --> C[Fetch HTML]
    C --> D[Parse Content]
    D --> E[Extract Text]
    E --> F[Document]
    
    style B fill:#FFD700
    style F fill:#90EE90
```

**Example:**
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=["https://example.com/article"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-content", "post-content")
        )
    )
)

docs = loader.load()
```

**File:** `2.Intermediate/1_web_loader.py`

---

### 2. JSON Loader

**Load structured JSON data**

```mermaid
graph TB
    A[JSON File] --> B[JSONLoader]
    B --> C{Parse Strategy}
    C -->|Full Document| D[Single Doc]
    C -->|Per Object| E[Multiple Docs]
    C -->|JSONPath| F[Filtered Docs]
    
    style B fill:#FFD700
```

**Example:**
```python
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[]",
    text_content=False
)

documents = loader.load()
```

**File:** `2.Intermediate/2_json_loader.py`

---

### 3. Markdown Loader

**Load and parse Markdown files**

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md")
docs = loader.load()
```

**File:** `2.Intermediate/3_markdown_loader.py`

---

### 4. API Loader

**Load data from REST APIs**

```mermaid
graph LR
    A[API Endpoint] --> B[APILoader]
    B --> C[Fetch Data]
    C --> D[Transform]
    D --> E[Documents]
    
    style B fill:#FFD700
```

**Example:**
```python
from langchain_community.document_loaders import APILoader

loader = APILoader(
    url="https://api.example.com/data",
    headers={"Authorization": "Bearer token"},
    jq_schema=".results[]"
)

documents = loader.load()
```

**File:** `2.Intermediate/4_api_loader.py`

---

### 5. YouTube Loader

**Extract transcripts from YouTube videos**

```python
from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    add_video_info=True
)

docs = loader.load()
```

**File:** `2.Intermediate/5_youtube_loader.py`

---

## Advanced Loaders

### 1. SQL Database Loader

**Query and load from databases**

```mermaid
graph TB
    A[SQL Query] --> B[SQLDatabaseLoader]
    B --> C[Execute Query]
    C --> D[Fetch Results]
    D --> E[Row 1: Document]
    D --> F[Row 2: Document]
    
    style B fill:#FFD700
```

**Example:**
```python
from langchain_community.document_loaders import SQLDatabaseLoader

loader = SQLDatabaseLoader(
    query="SELECT * FROM articles WHERE category='AI'",
    db_uri="postgresql://user:pass@localhost/mydb",
    page_content_columns=["title", "content"],
    metadata_columns=["author", "date", "category"]
)

documents = loader.load()
```

**File:** `3.Advanced/1_sql_loader.py`

---

### 2. Notion Loader

**Load from Notion workspace**

```python
from langchain_community.document_loaders import NotionDirectoryLoader

loader = NotionDirectoryLoader("path/to/notion/export")
docs = loader.load()
```

**File:** `3.Advanced/2_notion_loader.py`

---

### 3. S3 File Loader

**Load from AWS S3**

```python
from langchain_community.document_loaders import S3FileLoader

loader = S3FileLoader(
    bucket="my-bucket",
    key="documents/file.pdf"
)

docs = loader.load()
```

**File:** `3.Advanced/3_s3_loader.py`

---

### 4. Custom Loader

**Build your own loader**

```mermaid
graph TB
    A[Custom Data Source] --> B[Implement<br/>BaseLoader]
    B --> C[lazy_load method]
    B --> D[load method]
    C --> E[Documents]
    D --> E
    
    style B fill:#FFD700
    style E fill:#90EE90
```

**Example:**
```python
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import Iterator

class CustomAPILoader(BaseLoader):
    """Load data from custom API."""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents."""
        response = self._fetch_data()
        for item in response:
            yield Document(
                page_content=item['text'],
                metadata=item['metadata']
            )
    
    def _fetch_data(self):
        """Fetch data from API."""
        # Your API logic here
        pass

# Usage
loader = CustomAPILoader(
    api_endpoint="https://api.example.com",
    api_key="your_key"
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

**File:** `3.Advanced/4_custom_loader.py`

---

## Best Practices

### 1. Metadata Management

```mermaid
graph LR
    A[Load Document] --> B[Add Source]
    B --> C[Add Timestamp]
    C --> D[Add Category]
    D --> E[Add Custom Fields]
    
    style A fill:#FFE4B5
    style E fill:#90EE90
```

**Good Metadata:**
```python
metadata = {
    "source": "document.pdf",           # Where it came from
    "page": 1,                          # Page number
    "author": "John Doe",               # Who created it
    "date": "2024-01-01",              # When created
    "category": "research",             # Classification
    "tags": ["AI", "ML"],              # Keywords
    "doc_id": "12345"                  # Unique identifier
}
```

---

### 2. Error Handling

```python
from langchain_community.document_loaders import PyPDFLoader

def safe_load_pdf(file_path):
    """Load PDF with error handling."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            print(f"Warning: No content extracted from {file_path}")
            return []
        
        return documents
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

# Usage
docs = safe_load_pdf("document.pdf")
```

---

### 3. Batch Loading

```python
from langchain_community.document_loaders import DirectoryLoader
from concurrent.futures import ThreadPoolExecutor

def load_directory_parallel(path, glob_pattern="**/*.txt"):
    """Load directory in parallel for speed."""
    
    loader = DirectoryLoader(
        path,
        glob=glob_pattern,
        show_progress=True,
        use_multithreading=True,
        max_workers=4
    )
    
    return loader.load()

# Usage
docs = load_directory_parallel("documents/", "**/*.pdf")
print(f"Loaded {len(docs)} documents")
```

---

### 4. Memory Optimization

```python
from langchain_community.document_loaders import PyPDFLoader

def load_large_pdf_lazy(file_path):
    """Load large PDF lazily to save memory."""
    
    loader = PyPDFLoader(file_path)
    
    # Use lazy_load for large files
    for page in loader.lazy_load():
        # Process one page at a time
        yield page

# Usage
for page in load_large_pdf_lazy("large_document.pdf"):
    # Process page
    print(f"Processing page {page.metadata['page']}")
```

---

## Comparison Guide

### Loader Selection Matrix

| Data Source | Best Loader | Alternative | Complexity |
|-------------|-------------|-------------|------------|
| **Text Files** | TextLoader | DirectoryLoader | 🔧 |
| **PDFs** | PyPDFLoader | PyMuPDFLoader | 🔧🔧 |
| **Word Docs** | Docx2txtLoader | UnstructuredWordLoader | 🔧🔧 |
| **Web Pages** | WebBaseLoader | BeautifulSoupLoader | 🔧🔧 |
| **CSV** | CSVLoader | PandasDataFrameLoader | 🔧 |
| **JSON** | JSONLoader | - | 🔧🔧 |
| **Markdown** | UnstructuredMarkdownLoader | - | 🔧 |
| **YouTube** | YoutubeLoader | - | 🔧🔧 |
| **SQL** | SQLDatabaseLoader | - | 🔧🔧🔧 |
| **APIs** | Custom Loader | - | 🔧🔧🔧 |

---

### Performance Comparison

```mermaid
graph TD
    A[Loader Performance] --> B[Speed]
    A --> C[Memory]
    A --> D[Quality]
    
    B --> B1[Fast: Text, CSV]
    B --> B2[Medium: PDF, Web]
    B --> B3[Slow: OCR, Video]
    
    C --> C1[Low: Streaming]
    C --> C2[Medium: Standard]
    C --> C3[High: Batch Load]
    
    D --> D1[Basic: Text]
    D --> D2[Good: PDF]
    D --> D3[Best: Structured]
    
    style A fill:#FFD700
```

---

## Real-World Examples

### Example 1: Knowledge Base from PDFs

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all PDFs from directory
loader = DirectoryLoader(
    "knowledge_base/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)

documents = loader.load()
print(f"Loaded {len(documents)} pages from PDFs")

# Add custom metadata
for doc in documents:
    doc.metadata['source_type'] = 'pdf'
    doc.metadata['loaded_at'] = datetime.now().isoformat()
```

---

### Example 2: Web Documentation Scraper

```python
from langchain_community.document_loaders import SitemapLoader

# Load entire documentation site
loader = SitemapLoader(
    web_path="https://example.com/sitemap.xml",
    filter_urls=[r"^https://example.com/docs/.*"],
    parsing_function=custom_parsing_function
)

docs = loader.load()
```

---

### Example 3: Multi-Source Loader

```python
def load_multi_source(config):
    """Load from multiple sources."""
    all_docs = []
    
    # Load PDFs
    if 'pdf_dir' in config:
        pdf_loader = DirectoryLoader(
            config['pdf_dir'],
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        all_docs.extend(pdf_loader.load())
    
    # Load web pages
    if 'urls' in config:
        for url in config['urls']:
            web_loader = WebBaseLoader(url)
            all_docs.extend(web_loader.load())
    
    # Load database
    if 'db_uri' in config:
        db_loader = SQLDatabaseLoader(
            query=config['query'],
            db_uri=config['db_uri']
        )
        all_docs.extend(db_loader.load())
    
    return all_docs

# Usage
config = {
    'pdf_dir': 'documents/',
    'urls': ['https://example.com/page1', 'https://example.com/page2'],
    'db_uri': 'postgresql://localhost/mydb',
    'query': 'SELECT * FROM articles'
}

documents = load_multi_source(config)
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Empty documents** | No content extracted | Check file format, try different loader |
| **Encoding errors** | Wrong encoding | Specify encoding: `encoding='utf-8'` |
| **Memory issues** | Large files | Use `lazy_load()` instead of `load()` |
| **Slow loading** | Sequential processing | Enable `use_multithreading=True` |
| **Missing metadata** | Not extracted | Add custom metadata after loading |
| **File not found** | Wrong path | Check file path, use absolute paths |

---

## Next Steps

After mastering Document Loaders:

1. **Text Splitters** - Split documents into chunks
2. **Vector Stores** - Store embeddings
3. **Retrievers** - Query documents
4. **RAG Chains** - Build Q&A systems

---

## 🎯 Summary

**Key Takeaways:**

1. **Document Loaders convert raw data into LangChain Documents**
2. **Choose the right loader for your data source**
3. **Always add meaningful metadata**
4. **Use lazy loading for large files**
5. **Handle errors gracefully**
6. **Batch process when possible**

**Common Pipeline:**
```
Data Source → Loader → Documents → Splitter → Chunks → Embeddings → Vector Store
```

---

**Happy Loading! 🚀**

Remember: Good document loading is the foundation of great RAG systems!



