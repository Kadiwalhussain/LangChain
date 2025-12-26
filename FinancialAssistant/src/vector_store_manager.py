"""Vector store management using Chroma."""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage Chroma vector store for financial documents."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.embeddings = self._get_embeddings()
        self.vectorstore: Optional[Chroma] = None
    
    def _get_embeddings(self):
        """Get embedding model based on configuration."""
        if settings.embedding_provider == "openai":
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )
        elif settings.embedding_provider == "ollama":
            logger.info("Using Ollama embeddings")
            return OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url
            )
        else:  # huggingface
            logger.info("Using HuggingFace embeddings")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def initialize_store(self) -> Chroma:
        """
        Initialize or load existing vector store.
        
        Returns:
            Chroma vector store instance
        """
        try:
            self.vectorstore = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.vector_store_path
            )
            logger.info("Vector store initialized successfully")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not self.vectorstore:
            self.initialize_store()
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            ids = self.vectorstore.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} documents")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: dict = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            self.initialize_store()
        
        k = k or settings.top_k_results
        
        try:
            logger.info(f"Searching for: '{query}' (top {k} results)")
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: dict = None
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            self.initialize_store()
        
        k = k or settings.top_k_results
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            raise
    
    def get_retriever(self, **kwargs):
        """
        Get retriever interface.
        
        Returns:
            Retriever instance
        """
        if not self.vectorstore:
            self.initialize_store()
        
        search_kwargs = {
            "k": kwargs.get("k", settings.top_k_results),
        }
        
        if "filter" in kwargs:
            search_kwargs["filter"] = kwargs["filter"]
        
        return self.vectorstore.as_retriever(
            search_type=kwargs.get("search_type", "similarity"),
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
                logger.info("Collection deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
                raise
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.vectorstore:
            self.initialize_store()
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": settings.collection_name,
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = VectorStoreManager()
    manager.initialize_store()
    
    # Get stats
    stats = manager.get_collection_stats()
    print(f"Collection stats: {stats}")
