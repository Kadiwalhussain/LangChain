from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Embedding with Prompt Engineering
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand human language",
    "Computer vision enables machines to interpret visual information",
    "Reinforcement learning trains agents through rewards and penalties"
]

# Create embeddings and vector store
db = FAISS.from_texts(documents, embeddings)

# Prompt template for semantic search
search_prompt = PromptTemplate(
    input_variables=["query"],
    template="Search query: {query}"
)

# Perform semantic search
query = "What is deep learning?"
results = db.similarity_search(query, k=2)

print("Semantic Search Results:")
print(f"Query: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}\n")

print("="*50 + "\n")

# Prompt for RAG (Retrieval Augmented Generation)
from langchain.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
)

# Simulate RAG
context = "\n".join([doc.page_content for doc in results])
formatted_prompt = rag_prompt.format(context=context, question="Tell me about deep learning")
print("RAG Prompt Output:")
print(formatted_prompt)
