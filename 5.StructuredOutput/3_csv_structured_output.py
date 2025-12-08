"""
Basic Structured Output - Comma Separated Values (CSV)
Getting data in CSV format from LLM
"""

from langchain_ollama import OllamaLLM

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Prompt that asks for CSV output
prompt = """
List 5 programming languages with their year of creation in CSV format.
Return ONLY the CSV data with headers (no markdown, no extra text):

language,year_created
Python,1991
Java,1995
"""

# Get response
response = llm.invoke(prompt)

print("=" * 60)
print("CSV STRUCTURED OUTPUT")
print("=" * 60)

# Parse CSV
lines = response.strip().split('\n')

# Extract headers
headers = lines[0].split(',')
print(f"\n📋 Headers: {headers}")

# Parse data rows
print(f"\n📊 Data:")
data = []
for line in lines[1:]:
    if line.strip():  # Skip empty lines
        values = line.split(',')
        row = {headers[i]: values[i].strip() for i in range(len(headers))}
        data.append(row)
        print(f"  {row}")

print(f"\n✅ Total records: {len(data)}")
print("=" * 60)
