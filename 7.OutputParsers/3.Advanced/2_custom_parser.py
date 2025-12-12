"""
Advanced Output Parser #2: Custom Output Parser
Create your own parser for specific formats
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import BaseOutputParser
from typing import List

# Custom parser for parsing bullet points
class BulletPointParser(BaseOutputParser[List[str]]):
    """Parse bullet-pointed lists from LLM output"""
    
    def parse(self, text: str) -> List[str]:
        """Parse the output"""
        lines = text.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            # Handle different bullet formats
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Remove bullet and clean
                clean_line = line[1:].strip()
                if clean_line:
                    bullet_points.append(clean_line)
            elif line and line[0].isdigit() and '.' in line:
                # Handle numbered lists "1. Item"
                parts = line.split('.', 1)
                if len(parts) > 1:
                    clean_line = parts[1].strip()
                    if clean_line:
                        bullet_points.append(clean_line)
        
        return bullet_points
    
    def get_format_instructions(self) -> str:
        """Instructions for LLM"""
        return """Please format your response as a bullet-pointed list.
Use one of these formats:
• Item one
• Item two
OR
- Item one  
- Item two
OR
1. Item one
2. Item two"""

# Custom parser for key-value pairs
class KeyValueParser(BaseOutputParser[dict]):
    """Parse key-value pairs from LLM output"""
    
    def parse(self, text: str) -> dict:
        """Parse key-value pairs"""
        result = {}
        lines = text.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                result[key] = value
        
        return result
    
    def get_format_instructions(self) -> str:
        """Instructions for LLM"""
        return """Please format your response as key-value pairs:
Key1: Value1
Key2: Value2
Key3: Value3"""

# Initialize
llm = OllamaLLM(model="mistral")

print("=" * 80)
print("CUSTOM OUTPUT PARSERS")
print("=" * 80)

# Test BulletPointParser
print(f"\n1️⃣  BULLET POINT PARSER")
print("=" * 80)

bullet_parser = BulletPointParser()
prompt1 = f"""List 5 benefits of exercise.

{bullet_parser.get_format_instructions()}
"""

chain1 = llm | bullet_parser
result1 = chain1.invoke(prompt1)

print(f"\n✅ Parsed Bullet Points:")
print(f"   Type: {type(result1)}")
for i, item in enumerate(result1, 1):
    print(f"   {i}. {item}")

# Test KeyValueParser
print(f"\n2️⃣  KEY-VALUE PARSER")
print("=" * 80)

kv_parser = KeyValueParser()
prompt2 = f"""Provide information about Python programming language.

{kv_parser.get_format_instructions()}

Include: Name, Year Created, Creator, Type
"""

chain2 = llm | kv_parser
result2 = chain2.invoke(prompt2)

print(f"\n✅ Parsed Key-Value Pairs:")
print(f"   Type: {type(result2)}")
for key, value in result2.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 80)
print("CREATING CUSTOM PARSERS")
print("=" * 80)
print("""
To create a custom parser:

1. Inherit from BaseOutputParser[OutputType]
2. Implement parse(text: str) -> OutputType
3. Implement get_format_instructions() -> str
4. Add error handling if needed

When to create custom parsers:
✅ Unique output format
✅ Domain-specific parsing
✅ Legacy format compatibility
✅ Special validation rules
✅ Performance optimization

Example use cases:
- Parse tables
- Extract specific patterns
- Clean messy output
- Domain-specific formats
""")
print("=" * 80)
