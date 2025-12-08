"""
Basic Structured Output - YAML Format
Getting data in YAML format
"""

from langchain_ollama import OllamaLLM

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Prompt that asks for YAML output
prompt = """
Provide information about the Solar System in YAML format.
Return ONLY valid YAML (no markdown, no extra text):

name: Solar System
planets:
  - name: Mercury
    type: Rocky
  - name: Venus
    type: Rocky
"""

# Get response
response = llm.invoke(prompt)

print("=" * 60)
print("YAML STRUCTURED OUTPUT")
print("=" * 60)

# Simple YAML parsing (basic level)
def parse_yaml_simple(yaml_string):
    """Very basic YAML parser for demonstration"""
    data = {}
    current_key = None
    current_list = None
    
    lines = yaml_string.strip().split('\n')
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        
        # Check for list item
        if stripped.startswith('- '):
            if current_list is None:
                current_list = []
            item = stripped[2:].strip()
            # Try to parse as key: value
            if ':' in item:
                k, v = item.split(':', 1)
                current_list.append({k.strip(): v.strip()})
            else:
                current_list.append(item)
        
        # Check for key-value pair
        elif ':' in stripped:
            key, value = stripped.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if current_list is not None:
                data[current_key] = current_list
                current_list = None
            
            if value == '':
                current_key = key
            else:
                data[key] = value
    
    if current_list is not None:
        data[current_key] = current_list
    
    return data

# Parse YAML
try:
    data = parse_yaml_simple(response)
    print("\n✅ Successfully parsed YAML:")
    
    for key, value in data.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        print(f"  - {k}: {v}")
                else:
                    print(f"  - {item}")
        else:
            print(f"{key}: {value}")
    
except Exception as e:
    print(f"❌ YAML parsing failed: {e}")
    print(f"Raw response:\n{response}")

print("\n" + "=" * 60)
