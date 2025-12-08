from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Using HuggingFace Instruct embeddings for better semantic search
# Good for query-document pairs
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cpu"}
)

# Embed a single text with instruction
text = "Represent this sentence for semantic search:"
query = "How does embeddings work?"
embedded_query = embeddings.embed_query(query)

print(f"Query: {query}")
print(f"Embedding dimension: {len(embedded_query)}")
print(f"First 5 values: {embedded_query[:5]}")

# Embed documents
documents = [
    "LangChain is a framework for developing applications with LLMs",
    "Embeddings convert text into numerical vectors",
    "Vector databases store and search embeddings efficiently"
]
embedded_docs = embeddings.embed_documents(documents)

print(f"\nEmbedded {len(embedded_docs)} documents")
for i, emb in enumerate(embedded_docs):
    print(f"Document {i+1} embedding dimension: {len(emb)}")
