"""
Intermediate Text Splitter #3: HTML Text Splitter
Split HTML documents by headers and semantic structure
"""

from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

print("=" * 80)
print("HTML HEADER TEXT SPLITTER")
print("=" * 80)

# Example 1: Basic HTML Splitting
print("\n" + "=" * 80)
print("Example 1: Split HTML by Headers")
print("=" * 80)

html_string = """
<!DOCTYPE html>
<html>
<head>
    <title>LangChain Documentation</title>
</head>
<body>
    <h1>LangChain Framework</h1>
    <p>LangChain is a framework for developing applications powered by language models.</p>
    
    <h2>Core Components</h2>
    <p>The framework consists of several key components.</p>
    
    <h3>LLMs and Chat Models</h3>
    <p>These provide the text generation capabilities. LLMs handle completion tasks while
    chat models are optimized for conversational interfaces.</p>
    
    <h3>Prompt Templates</h3>
    <p>Templates help create reusable prompts with variables. They make it easy to
    structure inputs consistently.</p>
    
    <h2>Getting Started</h2>
    <p>Installation is simple using pip.</p>
    
    <h3>Installation</h3>
    <p>Run the following command:</p>
    <pre><code>pip install langchain</code></pre>
    
    <h3>Quick Example</h3>
    <p>Here's a minimal example to get started:</p>
    <pre><code>from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
response = llm.invoke("Hello!")</code></pre>
    
    <h2>Advanced Features</h2>
    <p>LangChain offers advanced capabilities for complex applications.</p>
    
    <h3>Chains</h3>
    <p>Chains connect multiple components for sophisticated workflows.</p>
    
    <h3>Agents</h3>
    <p>Agents can make decisions and use tools dynamically.</p>
</body>
</html>
"""

# Define headers to split on
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# Create HTML splitter
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Split the HTML
html_header_splits = html_splitter.split_text(html_string)

print(f"\n✅ Split HTML into {len(html_header_splits)} sections")
for i, doc in enumerate(html_header_splits, 1):
    print(f"\n--- Section {i} ---")
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content[:100]}...")

# Example 2: Web Scraping Use Case
print("\n" + "=" * 80)
print("Example 2: Processing Scraped Web Content")
print("=" * 80)

blog_html = """
<article>
    <h1>Introduction to Machine Learning</h1>
    <p>Machine learning is transforming industries worldwide.</p>
    <p>In this article, we'll explore the fundamentals.</p>
    
    <h2>Supervised Learning</h2>
    <p>Supervised learning uses labeled data to train models.</p>
    <p>Common algorithms include linear regression and decision trees.</p>
    
    <h3>Classification</h3>
    <p>Classification predicts discrete categories like spam/not spam.</p>
    <ul>
        <li>Logistic Regression</li>
        <li>Support Vector Machines</li>
        <li>Random Forests</li>
    </ul>
    
    <h3>Regression</h3>
    <p>Regression predicts continuous values like house prices.</p>
    <ul>
        <li>Linear Regression</li>
        <li>Polynomial Regression</li>
        <li>Ridge Regression</li>
    </ul>
    
    <h2>Unsupervised Learning</h2>
    <p>Unsupervised learning finds patterns in unlabeled data.</p>
    
    <h3>Clustering</h3>
    <p>Clustering groups similar data points together.</p>
    <p>K-means is a popular clustering algorithm.</p>
    
    <h3>Dimensionality Reduction</h3>
    <p>Reduces the number of features while preserving information.</p>
    <p>PCA is widely used for this purpose.</p>
</article>
"""

blog_splits = html_splitter.split_text(blog_html)

print(f"\n✅ Processed blog post into {len(blog_splits)} sections")
print("\nSection structure:")
for i, doc in enumerate(blog_splits, 1):
    headers = " > ".join([f"{v}" for k, v in doc.metadata.items()])
    print(f"\n{i}. {headers}")
    print(f"   Content: {doc.page_content.strip()[:80]}...")

# Example 3: Documentation Site
print("\n" + "=" * 80)
print("Example 3: API Documentation Processing")
print("=" * 80)

api_docs_html = """
<div class="documentation">
    <h1>API Reference</h1>
    <p>Complete API documentation for our service.</p>
    
    <h2>Authentication</h2>
    <p>All API requests require authentication using an API key.</p>
    <p>Include the key in the Authorization header.</p>
    
    <h2>Endpoints</h2>
    <p>Available API endpoints:</p>
    
    <h3>GET /users</h3>
    <p>Retrieve a list of users.</p>
    <p><strong>Parameters:</strong></p>
    <ul>
        <li>limit (optional): Number of results (default: 10)</li>
        <li>offset (optional): Pagination offset (default: 0)</li>
    </ul>
    <p><strong>Response:</strong> Array of user objects</p>
    
    <h3>POST /users</h3>
    <p>Create a new user.</p>
    <p><strong>Body:</strong></p>
    <pre>{"name": "John", "email": "john@example.com"}</pre>
    <p><strong>Response:</strong> Created user object</p>
    
    <h3>GET /users/:id</h3>
    <p>Retrieve a specific user by ID.</p>
    <p><strong>Parameters:</strong> User ID in URL</p>
    
    <h2>Error Codes</h2>
    <p>Standard HTTP status codes are used.</p>
    
    <h3>4xx Client Errors</h3>
    <p>400: Bad Request - Invalid parameters</p>
    <p>401: Unauthorized - Invalid API key</p>
    <p>404: Not Found - Resource doesn't exist</p>
    
    <h3>5xx Server Errors</h3>
    <p>500: Internal Server Error</p>
    <p>503: Service Unavailable</p>
</div>
"""

api_splits = html_splitter.split_text(api_docs_html)

print(f"\n✅ API docs split into {len(api_splits)} sections")
for i, doc in enumerate(api_splits, 1):
    print(f"\n--- Section {i} ---")
    print(f"Headers: {doc.metadata}")
    content_preview = doc.page_content.strip()[:100].replace("\n", " ")
    print(f"Content: {content_preview}...")

# Example 4: Combining with Size-Based Splitting
print("\n" + "=" * 80)
print("Example 4: Two-Stage Splitting (Headers + Size)")
print("=" * 80)

large_section_html = """
<html>
    <h1>Comprehensive Guide</h1>
    <p>This section contains a lot of content that might be too large for a single chunk.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
    exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
    
    <h2>Section One</h2>
    <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
    eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt 
    in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis 
    unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.</p>
    <p>Totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi 
    architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia 
    voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores.</p>
    
    <h2>Section Two</h2>
    <p>Another large section with lots of content that needs to be processed carefully.</p>
</html>
"""

# Stage 1: Split by HTML headers
header_chunks = html_splitter.split_text(large_section_html)

# Stage 2: Further split by size
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

final_chunks = char_splitter.split_documents(header_chunks)

print(f"\n✅ Two-stage splitting result:")
print(f"   Header-based splits: {len(header_chunks)}")
print(f"   Final size-based chunks: {len(final_chunks)}")
print("\nFinal chunks with preserved metadata:")
for i, chunk in enumerate(final_chunks[:3], 1):
    print(f"\n{i}. Headers: {chunk.metadata}")
    print(f"   Size: {len(chunk.page_content)} chars")
    print(f"   Content: {chunk.page_content[:80]}...")

# Example 5: Nested Structure
print("\n" + "=" * 80)
print("Example 5: Deeply Nested HTML Structure")
print("=" * 80)

nested_html = """
<html>
    <body>
        <h1>Main Topic</h1>
        <p>Introduction to the main topic.</p>
        
        <div class="section">
            <h2>Subtopic A</h2>
            <p>Content for subtopic A.</p>
            
            <div class="subsection">
                <h3>Detail A1</h3>
                <p>Detailed information about A1.</p>
                
                <h4>Point A1.1</h4>
                <p>Even more specific detail.</p>
            </div>
            
            <div class="subsection">
                <h3>Detail A2</h3>
                <p>Detailed information about A2.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Subtopic B</h2>
            <p>Content for subtopic B.</p>
        </div>
    </body>
</html>
"""

# Include h4 in splitting
nested_headers = [
    ("h1", "Title"),
    ("h2", "Section"),
    ("h3", "Subsection"),
    ("h4", "Detail"),
]

nested_splitter = HTMLHeaderTextSplitter(headers_to_split_on=nested_headers)
nested_splits = nested_splitter.split_text(nested_html)

print(f"\n✅ Nested HTML split into {len(nested_splits)} sections")
for i, doc in enumerate(nested_splits, 1):
    print(f"\n--- Section {i} ---")
    hierarchy = " > ".join([f"{v}" for k, v in doc.metadata.items()])
    print(f"Hierarchy: {hierarchy}")
    print(f"Content: {doc.page_content.strip()[:60]}...")

# Example 6: Real-world Wikipedia-style
print("\n" + "=" * 80)
print("Example 6: Wikipedia-Style Article Processing")
print("=" * 80)

wiki_html = """
<article>
    <h1>Artificial Intelligence</h1>
    <p>Artificial intelligence (AI) is intelligence demonstrated by machines.</p>
    
    <h2>History</h2>
    <p>The field was founded on the assumption that human intelligence can be 
    precisely described.</p>
    
    <h2>Applications</h2>
    <p>AI has been used in a wide range of fields.</p>
    
    <h3>Healthcare</h3>
    <p>AI is used for diagnosis and treatment planning.</p>
    
    <h3>Finance</h3>
    <p>Algorithmic trading and fraud detection use AI.</p>
    
    <h3>Transportation</h3>
    <p>Self-driving cars rely heavily on AI.</p>
    
    <h2>Ethics and Safety</h2>
    <p>AI raises important ethical questions about privacy and bias.</p>
</article>
"""

wiki_splits = html_splitter.split_text(wiki_html)

print(f"\n✅ Wikipedia article split into {len(wiki_splits)} knowledge chunks")
print("\nKnowledge base entries:")
for i, doc in enumerate(wiki_splits, 1):
    title = doc.metadata.get('Header 1', 'Unknown')
    section = doc.metadata.get('Header 2', 'Main')
    subsection = doc.metadata.get('Header 3', '')
    
    full_title = f"{title} - {section}"
    if subsection:
        full_title += f" - {subsection}"
    
    print(f"\n{i}. {full_title}")
    print(f"   {doc.page_content.strip()[:70]}...")

# Configuration guide
print("\n" + "=" * 80)
print("CONFIGURATION GUIDE")
print("=" * 80)
print("""
Basic HTML Splitting:

from langchain_text_splitters import HTMLHeaderTextSplitter

# Define headers to extract and split on
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),  # Optional: for deeply nested content
]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Split HTML content
chunks = splitter.split_text(html_string)

# Or split from URL
# chunks = splitter.split_text_from_url(url)

Combining with Size Control:

# Stage 1: Split by headers
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
header_chunks = html_splitter.split_text(html_content)

# Stage 2: Control size
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
   - Web scraping and indexing
   - Documentation sites
   - Blog post processing
   - Wiki article parsing
   - Knowledge base creation
   - API documentation
   - HTML email processing

🎯 Benefits:
   - Preserves HTML structure
   - Extracts hierarchical metadata
   - Cleans HTML tags automatically
   - Maintains semantic context
   - Easy integration with web scrapers
""")

print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Automatic HTML tag removal
   - Preserves document hierarchy
   - Extracts header structure
   - Great for web content
   - Handles nested headers

❌ Cons:
   - Only for HTML content
   - May need cleanup for messy HTML
   - Large sections require two-stage split
   - Depends on well-structured HTML

💡 Tip: Clean HTML with BeautifulSoup before splitting for best results
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print("""
from langchain_text_splitters import HTMLHeaderTextSplitter

# Basic usage
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "H1"),
        ("h2", "H2"),
        ("h3", "H3"),
    ]
)

# From string
chunks = splitter.split_text(html_string)

# From URL (requires requests library)
# chunks = splitter.split_text_from_url("https://example.com")

# Access metadata
for chunk in chunks:
    print(chunk.metadata)  # {'H1': '...', 'H2': '...'}
    print(chunk.page_content)  # Clean text without HTML tags
""")
