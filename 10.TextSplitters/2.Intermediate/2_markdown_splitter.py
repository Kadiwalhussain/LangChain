"""
Intermediate Text Splitter #2: Markdown Text Splitter
Split Markdown documents while preserving structure and hierarchy
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

print("=" * 80)
print("MARKDOWN TEXT SPLITTER")
print("=" * 80)

# Example 1: Basic Markdown Header Splitting
print("\n" + "=" * 80)
print("Example 1: Split by Markdown Headers")
print("=" * 80)

markdown_document = """
# LangChain Documentation

## Introduction

LangChain is a framework for developing applications powered by language models.

## Core Components

### LLMs and Chat Models

These are the foundation of LangChain. They provide the text generation capabilities.

### Prompt Templates

Templates help structure prompts with variables for reusability.

### Output Parsers

Parsers transform raw LLM outputs into structured data formats.

## Getting Started

### Installation

Install LangChain using pip:

```bash
pip install langchain
```

### Basic Example

Here's a simple example:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
response = llm.invoke("Hello!")
```

## Advanced Topics

### Chains

Chains connect multiple components together for complex workflows.

### Agents

Agents can make decisions and use tools to accomplish tasks.

## Conclusion

LangChain simplifies building LLM applications with modular components.
"""

# Define headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Create markdown splitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Split the document
md_header_splits = markdown_splitter.split_text(markdown_document)

print(f"\n✅ Split into {len(md_header_splits)} sections")
for i, doc in enumerate(md_header_splits, 1):
    print(f"\n--- Section {i} ---")
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content[:100]}...")

# Example 2: Hierarchical Structure Preservation
print("\n" + "=" * 80)
print("Example 2: Preserving Hierarchical Context")
print("=" * 80)

hierarchical_md = """
# User Guide

## Account Management

### Creating an Account

To create an account, follow these steps:
1. Visit the signup page
2. Enter your email
3. Choose a password

### Deleting an Account

To delete your account:
1. Go to settings
2. Click "Delete Account"
3. Confirm deletion

## Privacy Settings

### Data Collection

We collect minimal data to improve service.

### Data Sharing

We never share your personal information.
"""

md_splits = markdown_splitter.split_text(hierarchical_md)

print(f"\n✅ Split into {len(md_splits)} hierarchical sections")
for i, doc in enumerate(md_splits, 1):
    print(f"\n--- Section {i} ---")
    # Show hierarchical metadata
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    print(f"Content: {doc.page_content.strip()[:80]}...")

# Example 3: Combining with Character Splitter
print("\n" + "=" * 80)
print("Example 3: Two-Stage Splitting (Headers + Size)")
print("=" * 80)

long_section_md = """
# Data Processing Guide

## Introduction

Data processing is a critical step in any machine learning pipeline. It involves
cleaning, transforming, and preparing raw data for analysis. This guide covers
the essential techniques and best practices for effective data processing.

The process typically includes several stages: data collection, cleaning,
transformation, validation, and storage. Each stage has its own challenges
and requires specific tools and techniques.

## Data Collection

### Sources

Data can come from various sources including databases, APIs, files, and streams.
Each source has different characteristics and requires different handling.

Databases provide structured data with schemas. APIs offer programmatic access
to external services. Files can contain structured or unstructured data.
Streams provide real-time data flow.

### Best Practices

Always validate data at the point of collection. Implement error handling for
failed requests. Use appropriate data types for storage. Document your data
sources and collection methods.
"""

# First split by headers
header_splits = markdown_splitter.split_text(long_section_md)

# Then split large sections by size
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

# Process each section
final_chunks = []
for doc in header_splits:
    # If section is too large, split it further
    if len(doc.page_content) > 200:
        chunks = char_splitter.split_text(doc.page_content)
        for chunk in chunks:
            # Create new document preserving metadata
            from langchain_core.documents import Document
            final_chunks.append(
                Document(page_content=chunk, metadata=doc.metadata)
            )
    else:
        final_chunks.append(doc)

print(f"\n✅ Two-stage splitting result:")
print(f"   Header splits: {len(header_splits)}")
print(f"   Final chunks: {len(final_chunks)}")
print("\nFinal chunks with metadata:")
for i, chunk in enumerate(final_chunks[:3], 1):
    print(f"\n{i}. Metadata: {chunk.metadata}")
    print(f"   Content: {chunk.page_content[:80]}...")

# Example 4: README Processing
print("\n" + "=" * 80)
print("Example 4: Processing README Files")
print("=" * 80)

readme_content = """
# Awesome Project

A brief description of the project.

## Features

- Feature 1: Does something cool
- Feature 2: Does something else
- Feature 3: Another great feature

## Installation

```bash
npm install awesome-project
```

## Usage

```javascript
import { AwesomeProject } from 'awesome-project';

const project = new AwesomeProject();
project.run();
```

## API Reference

### AwesomeProject

Main class for the project.

#### Methods

- `run()`: Execute the project
- `stop()`: Stop execution

## Contributing

Pull requests are welcome!

## License

MIT License
"""

readme_splits = markdown_splitter.split_text(readme_content)

print(f"\n✅ README split into {len(readme_splits)} sections")
print("\nSection structure:")
for i, doc in enumerate(readme_splits, 1):
    headers = [f"{k}: {v}" for k, v in doc.metadata.items()]
    print(f"{i}. {' > '.join(headers) if headers else 'No headers'}")
    print(f"   Length: {len(doc.page_content)} chars")

# Example 5: Code Block Handling
print("\n" + "=" * 80)
print("Example 5: Handling Code Blocks in Markdown")
print("=" * 80)

code_heavy_md = """
# Python Tutorial

## Variables

Variables store data:

```python
name = "Alice"
age = 30
```

## Functions

Functions encapsulate logic:

```python
def greet(name):
    return f"Hello, {name}!"

result = greet("Bob")
print(result)
```

## Classes

Classes define objects:

```python
class Person:
    def __init__(self, name):
        self.name = name
    
    def introduce(self):
        return f"I am {self.name}"
```
"""

code_splits = markdown_splitter.split_text(code_heavy_md)

print(f"\n✅ Split code-heavy markdown into {len(code_splits)} sections")
for i, doc in enumerate(code_splits, 1):
    has_code = "```" in doc.page_content
    print(f"\nSection {i} - {doc.metadata}")
    print(f"   Contains code: {'Yes' if has_code else 'No'}")
    print(f"   Length: {len(doc.page_content)} chars")

# Example 6: Custom Header Handling
print("\n" + "=" * 80)
print("Example 6: Custom Header Configuration")
print("=" * 80)

# Only split on top-level headers
top_level_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Chapter")]
)

# Split on all levels
all_levels_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Topic")
    ]
)

sample_md = """
# Chapter 1
Content for chapter 1.

## Section 1.1
Content for section 1.1.

### Subsection 1.1.1
Detailed content.

# Chapter 2
Content for chapter 2.
"""

top_level_splits = top_level_splitter.split_text(sample_md)
all_level_splits = all_levels_splitter.split_text(sample_md)

print(f"\n📊 Comparison:")
print(f"   Top-level only: {len(top_level_splits)} chunks")
print(f"   All levels: {len(all_level_splits)} chunks")

# Configuration guide
print("\n" + "=" * 80)
print("CONFIGURATION GUIDE")
print("=" * 80)
print("""
Basic Markdown Splitting:

from langchain_text_splitters import MarkdownHeaderTextSplitter

# Define headers to split on
headers_to_split_on = [
    ("#", "Header 1"),      # H1
    ("##", "Header 2"),     # H2
    ("###", "Header 3"),    # H3
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep headers in content
)

# Split text
chunks = splitter.split_text(markdown_text)

Two-Stage Splitting (for large sections):

# Stage 1: Split by headers
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
header_chunks = md_splitter.split_text(markdown_text)

# Stage 2: Further split large chunks
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

final_chunks = char_splitter.split_documents(header_chunks)
""")

print("\n" + "=" * 80)
print("USE CASES")
print("=" * 80)
print("""
✅ Perfect for:
   - Documentation processing
   - README files
   - Technical wikis
   - Blog posts
   - Knowledge bases
   - API documentation
   - Tutorial content

🎯 Benefits:
   - Preserves document structure
   - Maintains header hierarchy
   - Keeps related content together
   - Metadata includes context
   - Natural semantic boundaries
""")

print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Structure-aware splitting
   - Preserves hierarchy in metadata
   - Natural semantic boundaries
   - Great for documentation
   - Easy to configure

❌ Cons:
   - Only for Markdown format
   - May create very large chunks
   - Requires well-structured documents
   - Need two-stage split for size control

💡 Tip: Combine with RecursiveCharacterTextSplitter for size management
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print("""
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Create splitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
)

# Split markdown
chunks = splitter.split_text(markdown_text)

# Access metadata
for chunk in chunks:
    print(chunk.metadata)  # {'H1': '...', 'H2': '...'}
    print(chunk.page_content)
""")
