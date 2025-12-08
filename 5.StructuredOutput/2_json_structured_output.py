"""
Basic Structured Output - Dictionary/JSON Response
Here we ask the LLM to return data in a specific JSON format
"""

import json
from langchain_ollama import OllamaLLM

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Prompt that asks for JSON output
prompt = """
Extract information about Paris in JSON format. 
Return ONLY valid JSON (no other text) with these keys:
- city_name
- country
- population
- famous_landmark

Here's an example format:
{
  "city_name": "Paris",
  "country": "France",
  "population": "2.2 million",
  "famous_landmark": "Eiffel Tower"
}
"""

# Get response
response = llm.invoke(prompt)

print("=" * 60)
print("JSON STRUCTURED OUTPUT")
print("=" * 60)

# Try to parse JSON
try:
    # Clean up response if needed (remove markdown code blocks)
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    
    data = json.loads(cleaned)
    
    print("\n✅ Successfully parsed JSON:")
    print(json.dumps(data, indent=2))
    
    print("\n📋 Accessing individual fields:")
    print(f"City: {data['city_name']}")
    print(f"Country: {data['country']}")
    print(f"Population: {data['population']}")
    print(f"Landmark: {data['famous_landmark']}")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
    print(f"Raw response: {response}")

print("=" * 60)
