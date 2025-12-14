"""
Intermediate Text Splitter #1: Language-Specific Splitters
Split code by respecting programming language syntax
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

print("=" * 80)
print("LANGUAGE-SPECIFIC TEXT SPLITTERS")
print("=" * 80)

# Example 1: Python Code Splitting
print("\n" + "=" * 80)
print("Example 1: Python Code Splitting")
print("=" * 80)

python_code = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    """Process and analyze data."""
    
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        """Process the data."""
        self.processed = True
        return [x * 2 for x in self.data]
    
    def analyze(self):
        """Analyze processed data."""
        if not self.processed:
            raise ValueError("Data not processed yet")
        return sum(self.data) / len(self.data)

def main():
    """Main function."""
    numbers = [1, 2, 3, 4, 5]
    processor = DataProcessor(numbers)
    result = processor.process()
    print(f"Processed: {result}")
    print(f"Analysis: {processor.analyze()}")

if __name__ == "__main__":
    main()
'''

# Create Python splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=30
)

python_chunks = python_splitter.split_text(python_code)

print(f"\n✅ Split Python code into {len(python_chunks)} chunks")
for i, chunk in enumerate(python_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("-" * 40)

# Example 2: JavaScript Code Splitting
print("\n" + "=" * 80)
print("Example 2: JavaScript Code Splitting")
print("=" * 80)

javascript_code = '''
// User authentication service
class AuthService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.token = null;
    }
    
    async login(username, password) {
        const response = await fetch(`${this.apiUrl}/login`, {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
        
        if (response.ok) {
            const data = await response.json();
            this.token = data.token;
            return true;
        }
        return false;
    }
    
    async getUserData() {
        if (!this.token) {
            throw new Error('Not authenticated');
        }
        
        const response = await fetch(`${this.apiUrl}/user`, {
            headers: {
                'Authorization': `Bearer ${this.token}`
            }
        });
        
        return await response.json();
    }
}

// Helper functions
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US');
}

// Initialize service
const authService = new AuthService('https://api.example.com');
'''

# Create JavaScript splitter
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=250,
    chunk_overlap=30
)

js_chunks = js_splitter.split_text(javascript_code)

print(f"\n✅ Split JavaScript code into {len(js_chunks)} chunks")
for i, chunk in enumerate(js_chunks[:2], 1):  # Show first 2
    print(f"\nChunk {i}:")
    print(chunk)
    print("-" * 40)

# Example 3: Supported Languages
print("\n" + "=" * 80)
print("Example 3: Supported Programming Languages")
print("=" * 80)

supported_languages = [
    ("Python", Language.PYTHON),
    ("JavaScript", Language.JS),
    ("TypeScript", Language.TS),
    ("Java", Language.JAVA),
    ("C", Language.C),
    ("C++", Language.CPP),
    ("C#", Language.CSHARP),
    ("Go", Language.GO),
    ("Rust", Language.RUST),
    ("Ruby", Language.RUBY),
    ("PHP", Language.PHP),
    ("Swift", Language.SWIFT),
    ("Kotlin", Language.KOTLIN),
    ("Scala", Language.SCALA),
    ("HTML", Language.HTML),
    ("Markdown", Language.MARKDOWN),
]

print("\n📋 Available Language Splitters:")
for name, lang in supported_languages:
    print(f"   - {name:<15} Language.{lang.value.upper()}")

# Example 4: Comparison with regular splitter
print("\n" + "=" * 80)
print("Example 4: Language Splitter vs Regular Splitter")
print("=" * 80)

comparison_code = '''
def process_data(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result

def analyze_data(data):
    return sum(data) / len(data)
'''

# Regular splitter
regular_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)

# Language-aware splitter
lang_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=80,
    chunk_overlap=10
)

regular_chunks = regular_splitter.split_text(comparison_code)
lang_chunks = lang_splitter.split_text(comparison_code)

print("\n📌 Regular Splitter (may break syntax):")
for i, chunk in enumerate(regular_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

print("\n📌 Language-Aware Splitter (respects syntax):")
for i, chunk in enumerate(lang_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

# Example 5: Real-world use case - Code documentation
print("\n" + "=" * 80)
print("Example 5: Code Documentation Generation")
print("=" * 80)

code_for_docs = '''
class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a, b):
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''

doc_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,
    chunk_overlap=20
)

doc_chunks = doc_splitter.split_text(code_for_docs)

print(f"\n✅ Split code into {len(doc_chunks)} documentation chunks")
print("\nEach chunk can be sent to LLM for documentation:")
for i, chunk in enumerate(doc_chunks, 1):
    print(f"\n--- Chunk {i} (for documentation) ---")
    print(chunk)

# Example 6: Multi-language project
print("\n" + "=" * 80)
print("Example 6: Processing Multi-Language Projects")
print("=" * 80)

# Simulate processing different file types
files = {
    "app.py": (Language.PYTHON, "def main():\n    print('Hello')\n"),
    "script.js": (Language.JS, "function hello() {\n    console.log('Hello');\n}\n"),
    "README.md": (Language.MARKDOWN, "# Project\n\nDescription here\n"),
}

print("\n📁 Processing project files:")
for filename, (language, content) in files.items():
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = splitter.split_text(content)
    print(f"\n{filename} ({language.value}):")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Preview: {content[:50]}...")

# Configuration guide
print("\n" + "=" * 80)
print("CONFIGURATION GUIDE")
print("=" * 80)
print("""
Language-Specific Settings:

1. Python Code:
   from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
   
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON,
       chunk_size=1000,
       chunk_overlap=200
   )

2. JavaScript/TypeScript:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.JS,  # or Language.TS
       chunk_size=1000,
       chunk_overlap=200
   )

3. Markdown Documentation:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.MARKDOWN,
       chunk_size=1000,
       chunk_overlap=100
   )

💡 The splitter automatically uses appropriate separators for each language!
""")

print("\n" + "=" * 80)
print("USE CASES")
print("=" * 80)
print("""
✅ Perfect for:
   - Code documentation generation
   - Code search and indexing
   - Repository analysis
   - Code review automation
   - Technical documentation
   - Multi-file project processing

🎯 Benefits:
   - Respects language syntax
   - Keeps functions/classes together
   - Preserves code structure
   - Better context for LLMs
   - Reduces broken syntax
""")

print("\n" + "=" * 80)
print("PROS & CONS")
print("=" * 80)
print("""
✅ Pros:
   - Syntax-aware splitting
   - Preserves code structure
   - Better for code analysis
   - Supports many languages
   - Maintains context

❌ Cons:
   - Only for supported languages
   - May need language detection
   - Slightly more complex setup

🌟 Best Practice: Always use language-specific splitters for code!
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)
print("""
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Create language-specific splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # or JS, JAVA, etc.
    chunk_size=1000,
    chunk_overlap=200
)

# Split code
chunks = splitter.split_text(code_string)

# For documents
documents = splitter.create_documents([code_string])
""")
