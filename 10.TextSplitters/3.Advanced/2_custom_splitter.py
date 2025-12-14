"""
Advanced Text Splitter #2: Custom Text Splitter
Create your own text splitter for unique requirements
"""

from langchain_text_splitters import TextSplitter
from typing import List
import re

print("=" * 80)
print("CUSTOM TEXT SPLITTER - BUILD YOUR OWN")
print("=" * 80)

# Example 1: Basic Custom Splitter
print("\n" + "=" * 80)
print("Example 1: Creating a Simple Custom Splitter")
print("=" * 80)

class SimpleCustomSplitter(TextSplitter):
    """Split text by a custom delimiter."""
    
    def __init__(self, delimiter: str = "---", **kwargs):
        super().__init__(**kwargs)
        self.delimiter = delimiter
    
    def split_text(self, text: str) -> List[str]:
        """Split text by delimiter."""
        chunks = text.split(self.delimiter)
        # Clean up chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

# Test the custom splitter
sample_text = """
Section 1 content here.
This is the first section.
---
Section 2 content here.
This is the second section.
---
Section 3 content here.
This is the third section.
"""

custom_splitter = SimpleCustomSplitter(delimiter="---")
chunks = custom_splitter.split_text(sample_text)

print(f"\n✅ Custom delimiter splitter created {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

# Example 2: Bullet Point Splitter
print("\n" + "=" * 80)
print("Example 2: Bullet Point Text Splitter")
print("=" * 80)

class BulletPointSplitter(TextSplitter):
    """Split text by bullet points."""
    
    def split_text(self, text: str) -> List[str]:
        """Split on bullet points (-, •, *, etc.)."""
        # Split by various bullet point markers
        pattern = r'(?:^|\n)[\s]*[•\-\*]\s+'
        chunks = re.split(pattern, text)
        
        # Clean and filter
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

bullet_text = """
Key Features:
• Feature one with detailed description
• Feature two with more information
• Feature three with additional context
• Feature four completing the list
"""

bullet_splitter = BulletPointSplitter()
bullet_chunks = bullet_splitter.split_text(bullet_text)

print(f"\n✅ Bullet point splitter created {len(bullet_chunks)} chunks")
for i, chunk in enumerate(bullet_chunks, 1):
    print(f"\n{i}. {chunk}")

# Example 3: Paragraph Number Splitter
print("\n" + "=" * 80)
print("Example 3: Numbered Section Splitter")
print("=" * 80)

class NumberedSectionSplitter(TextSplitter):
    """Split text by numbered sections (1., 2., 3., etc.)."""
    
    def split_text(self, text: str) -> List[str]:
        """Split by numbered sections."""
        # Match patterns like "1.", "2.", etc. at start of line
        pattern = r'(?:^|\n)(\d+\.)\s+'
        
        # Split and keep the numbers
        parts = re.split(pattern, text)
        
        # Reconstruct chunks with their numbers
        chunks = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                number = parts[i]
                content = parts[i + 1].strip()
                if content:
                    chunks.append(f"{number} {content}")
        
        return chunks

numbered_text = """
Introduction to the topic.

1. First main point
This explains the first concept in detail.

2. Second main point
This covers the second important aspect.

3. Third main point
This discusses the final key element.

Conclusion wrapping up.
"""

numbered_splitter = NumberedSectionSplitter()
numbered_chunks = numbered_splitter.split_text(numbered_text)

print(f"\n✅ Numbered section splitter created {len(numbered_chunks)} chunks")
for chunk in numbered_chunks:
    print(f"\n{chunk}")

# Example 4: Time-based Splitter (for transcripts)
print("\n" + "=" * 80)
print("Example 4: Timestamp-Based Splitter (Transcript)")
print("=" * 80)

class TimestampSplitter(TextSplitter):
    """Split transcript by timestamps."""
    
    def split_text(self, text: str) -> List[str]:
        """Split by timestamp markers [HH:MM:SS]."""
        # Match timestamp pattern
        pattern = r'\[(\d{2}:\d{2}:\d{2})\]'
        
        # Split by timestamps
        parts = re.split(pattern, text)
        
        # Combine timestamps with content
        chunks = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                timestamp = parts[i]
                content = parts[i + 1].strip()
                if content:
                    chunks.append(f"[{timestamp}] {content}")
        
        return chunks

transcript = """
[00:00:15] Welcome to today's presentation on artificial intelligence.
[00:01:30] Let's start with the basics of machine learning.
[00:03:45] Neural networks are a key component of modern AI systems.
[00:05:20] Now let's discuss practical applications in industry.
[00:07:00] Thank you for your attention. Questions?
"""

timestamp_splitter = TimestampSplitter()
timestamp_chunks = timestamp_splitter.split_text(transcript)

print(f"\n✅ Timestamp splitter created {len(timestamp_chunks)} chunks")
for chunk in timestamp_chunks:
    print(f"\n{chunk}")

# Example 5: Advanced - Size-Aware Custom Splitter
print("\n" + "=" * 80)
print("Example 5: Size-Aware Custom Splitter")
print("=" * 80)

class SizeAwareCustomSplitter(TextSplitter):
    """Custom splitter that respects size limits."""
    
    def __init__(self, delimiter: str = "---", chunk_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.delimiter = delimiter
        self._chunk_size = chunk_size
    
    def split_text(self, text: str) -> List[str]:
        """Split by delimiter, then by size if needed."""
        # First split by delimiter
        sections = text.split(self.delimiter)
        
        final_chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is small enough, keep it
            if len(section) <= self._chunk_size:
                final_chunks.append(section)
            else:
                # Split large sections by sentences
                sentences = re.split(r'(?<=[.!?])\s+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self._chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
        
        return final_chunks

size_aware_text = """
Short section.
---
This is a much longer section that exceeds the chunk size limit. It contains multiple sentences. We need to split this intelligently. Each sentence should be preserved. The splitter will handle this automatically.
---
Another short section.
"""

size_aware_splitter = SizeAwareCustomSplitter(delimiter="---", chunk_size=100)
size_aware_chunks = size_aware_splitter.split_text(size_aware_text)

print(f"\n✅ Size-aware splitter created {len(size_aware_chunks)} chunks")
for i, chunk in enumerate(size_aware_chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk)

# Example 6: Domain-Specific Splitter (Medical Records)
print("\n" + "=" * 80)
print("Example 6: Domain-Specific Splitter (Medical Records)")
print("=" * 80)

class MedicalRecordSplitter(TextSplitter):
    """Split medical records by sections."""
    
    def split_text(self, text: str) -> List[str]:
        """Split by medical record sections."""
        # Common medical record section headers
        section_headers = [
            r'CHIEF COMPLAINT:',
            r'HISTORY OF PRESENT ILLNESS:',
            r'PAST MEDICAL HISTORY:',
            r'MEDICATIONS:',
            r'ALLERGIES:',
            r'PHYSICAL EXAMINATION:',
            r'ASSESSMENT:',
            r'PLAN:',
        ]
        
        # Create pattern that matches any header
        pattern = '(' + '|'.join(section_headers) + ')'
        
        # Split by headers, keeping them
        parts = re.split(pattern, text)
        
        # Combine headers with their content
        chunks = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                header = parts[i].strip()
                content = parts[i + 1].strip()
                if content:
                    chunks.append(f"{header}\n{content}")
        
        return chunks

medical_record = """
CHIEF COMPLAINT:
Patient reports chest pain for 2 hours.

HISTORY OF PRESENT ILLNESS:
55-year-old male with sudden onset chest pain while at rest.

PAST MEDICAL HISTORY:
Hypertension, Type 2 Diabetes, Hyperlipidemia.

MEDICATIONS:
Lisinopril 10mg daily, Metformin 500mg twice daily.

PHYSICAL EXAMINATION:
BP 145/90, HR 88, RR 16. Cardiovascular exam unremarkable.

ASSESSMENT:
Possible acute coronary syndrome.

PLAN:
ECG, cardiac enzymes, cardiology consult.
"""

medical_splitter = MedicalRecordSplitter()
medical_chunks = medical_splitter.split_text(medical_record)

print(f"\n✅ Medical record splitter created {len(medical_chunks)} sections")
for i, chunk in enumerate(medical_chunks, 1):
    print(f"\n--- Section {i} ---")
    print(chunk)

# Example 7: JSON-like Structure Splitter
print("\n" + "=" * 80)
print("Example 7: Structured Data Splitter")
print("=" * 80)

class StructuredDataSplitter(TextSplitter):
    """Split structured data by logical blocks."""
    
    def split_text(self, text: str) -> List[str]:
        """Split by balanced braces/brackets."""
        chunks = []
        current_chunk = ""
        brace_count = 0
        bracket_count = 0
        
        for char in text:
            current_chunk += char
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            
            # When all braces/brackets are closed
            if brace_count == 0 and bracket_count == 0 and current_chunk.strip():
                if current_chunk.strip() not in ['{', '}', '[', ']']:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

structured_text = """
{id: 1, name: "Alice", role: "Engineer"}
{id: 2, name: "Bob", role: "Designer"}
[task: "Review PR", assigned: "Alice"]
"""

structured_splitter = StructuredDataSplitter()
structured_chunks = structured_splitter.split_text(structured_text)

print(f"\n✅ Structured data splitter created {len(structured_chunks)} chunks")
for i, chunk in enumerate(structured_chunks, 1):
    print(f"\nChunk {i}: {chunk}")

# Configuration Guide
print("\n" + "=" * 80)
print("CREATING CUSTOM SPLITTERS - GUIDE")
print("=" * 80)

print("""
Step 1: Inherit from TextSplitter

from langchain_text_splitters import TextSplitter
from typing import List

class MyCustomSplitter(TextSplitter):
    pass

Step 2: Implement split_text method

def split_text(self, text: str) -> List[str]:
    \"\"\"Your custom splitting logic.\"\"\"
    # Your code here
    chunks = []
    # ... splitting logic ...
    return chunks

Step 3: Optional - Add custom initialization

def __init__(self, custom_param: str, **kwargs):
    super().__init__(**kwargs)
    self.custom_param = custom_param

Step 4: Use your splitter

splitter = MyCustomSplitter(custom_param="value")
chunks = splitter.split_text(text)

# Or create documents
docs = splitter.create_documents([text])

Complete Example:

class CustomSplitter(TextSplitter):
    def __init__(self, delimiter: str = "---", **kwargs):
        super().__init__(**kwargs)
        self.delimiter = delimiter
    
    def split_text(self, text: str) -> List[str]:
        chunks = text.split(self.delimiter)
        return [c.strip() for c in chunks if c.strip()]

splitter = CustomSplitter(delimiter="###")
chunks = splitter.split_text(text)
""")

# Use cases
print("\n" + "=" * 80)
print("WHEN TO CREATE CUSTOM SPLITTERS")
print("=" * 80)

print("""
✅ Create custom splitters for:

1. Domain-Specific Formats:
   - Legal documents (sections, clauses)
   - Medical records (standardized sections)
   - Financial reports (specific structure)
   - Scientific papers (IMRaD structure)

2. Proprietary Formats:
   - Company-specific document structure
   - Internal data formats
   - Legacy system outputs

3. Special Requirements:
   - Preserve specific delimiters
   - Complex splitting logic
   - Multi-stage processing
   - Custom metadata extraction

4. Performance Optimization:
   - Faster than general splitters
   - Optimized for your data
   - Reduced overhead

❌ Don't create custom splitters when:
   - Standard splitters work fine
   - Over-engineering simple tasks
   - Maintenance overhead not justified
   - Built-in splitters are better tested
""")

# Best practices
print("\n" + "=" * 80)
print("BEST PRACTICES")
print("=" * 80)

print("""
1. Start Simple:
   ✅ Begin with basic splitting logic
   ✅ Add complexity incrementally
   ✅ Test thoroughly at each step

2. Handle Edge Cases:
   ✅ Empty strings
   ✅ Very long chunks
   ✅ Missing delimiters
   ✅ Invalid input

3. Document Your Splitter:
   ✅ Clear docstrings
   ✅ Usage examples
   ✅ Parameter descriptions

4. Make it Reusable:
   ✅ Configurable parameters
   ✅ Sensible defaults
   ✅ Clean interface

5. Test Thoroughly:
   ✅ Unit tests
   ✅ Edge cases
   ✅ Real-world data
   ✅ Performance tests
""")

# Quick reference
print("\n" + "=" * 80)
print("QUICK REFERENCE")
print("=" * 80)

print("""
from langchain_text_splitters import TextSplitter
from typing import List

class MyCustomSplitter(TextSplitter):
    \"\"\"Custom text splitter.\"\"\"
    
    def __init__(self, param: str, **kwargs):
        super().__init__(**kwargs)
        self.param = param
    
    def split_text(self, text: str) -> List[str]:
        \"\"\"Split text using custom logic.\"\"\"
        # Your custom splitting logic
        chunks = []
        # ... process text ...
        return chunks

# Usage
splitter = MyCustomSplitter(param="value")
chunks = splitter.split_text(text)

# Or with documents
docs = splitter.create_documents([text1, text2])
""")

print("\n" + "=" * 80)
print("🌟 REMEMBER")
print("=" * 80)
print("""
Custom splitters give you full control but:
- Require more development time
- Need thorough testing
- Must handle edge cases
- Require ongoing maintenance

Only create custom splitters when standard ones don't meet your needs!
""")
