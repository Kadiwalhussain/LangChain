from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Using OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single text
text = "LangChain is a framework for developing applications powered by language models"
embedded_text = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Embedding dimension: {len(embedded_text)}")
print(f"First 5 values: {embedded_text[:5]}")

# Embed multiple texts
texts = [
    "LangChain is great",
    "Embeddings are useful",
    "Vector databases store embeddings"
]
embedded_texts = embeddings.embed_documents(texts)

print(f"\nEmbedded {len(embedded_texts)} documents")
for i, emb in enumerate(embedded_texts):
    print(f"Document {i+1} embedding dimension: {len(emb)}")
