"""Question-Answering chain for financial queries."""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
import logging
from config import settings
from src.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class FinancialQAChain:
    """Q&A system for financial documents."""
    
    def __init__(self):
        """Initialize QA chain."""
        self.llm = self._get_llm()
        self.vector_manager = VectorStoreManager()
        self.vector_manager.initialize_store()
        self.qa_chain = self._create_qa_chain()
    
    def _get_llm(self):
        """Get LLM based on configuration."""
        if settings.llm_provider == "openai":
            logger.info("Using OpenAI LLM")
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.temperature,
                openai_api_key=settings.openai_api_key
            )
        else:  # ollama
            logger.info("Using Ollama LLM")
            return ChatOllama(
                model=settings.ollama_model,
                temperature=settings.temperature,
                base_url=settings.ollama_base_url
            )
    
    def _create_qa_chain(self):
        """Create the QA retrieval chain."""
        # Custom prompt for financial questions
        template = """You are a knowledgeable financial assistant specializing in tax and accounting matters.
Use the following context from financial documents to answer the question.

IMPORTANT GUIDELINES:
- Provide accurate, specific answers based on the context
- If you're not sure, say so - don't make up information
- Cite relevant sections or form numbers when applicable
- Explain financial terms in simple language
- Mention if professional advice is recommended

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the available information."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Get retriever
        retriever = self.vector_manager.get_retriever()
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about financial documents.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"Processing question: {question}")
            
            result = self.qa_chain.invoke({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result['result'],
                "sources": [
                    {
                        "file": doc.metadata.get('file_name', 'Unknown'),
                        "category": doc.metadata.get('category', 'Unknown'),
                        "page": doc.metadata.get('page', 'N/A')
                    }
                    for doc in result['source_documents']
                ]
            }
            
            logger.info("Question answered successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error: {str(e)}. Please try again.",
                "sources": []
            }
    
    def ask_with_filter(self, question: str, category: str = None) -> Dict[str, Any]:
        """
        Ask a question with document category filter.
        
        Args:
            question: User's question
            category: Document category to filter by
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Create filtered retriever
            filter_dict = {"category": category} if category else None
            retriever = self.vector_manager.get_retriever(filter=filter_dict)
            
            # Temporary chain with filtered retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": question})
            
            return {
                "question": question,
                "answer": result['result'],
                "category_filter": category,
                "sources": [
                    {
                        "file": doc.metadata.get('file_name', 'Unknown'),
                        "category": doc.metadata.get('category', 'Unknown')
                    }
                    for doc in result['source_documents']
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in filtered query: {e}")
            raise
    
    def get_relevant_context(self, question: str, k: int = 3):
        """
        Get relevant document chunks without generating an answer.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks
        """
        try:
            docs = self.vector_manager.similarity_search(question, k=k)
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []


# Standalone function for quick queries
def quick_ask(question: str) -> str:
    """
    Quick question asking without creating a full QA instance.
    
    Args:
        question: Question to ask
        
    Returns:
        Answer string
    """
    qa = FinancialQAChain()
    result = qa.ask(question)
    return result['answer']


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    qa = FinancialQAChain()
    
    # Example questions
    questions = [
        "What deductions can I claim for a home office?",
        "How do I report freelance income?",
        "What is the standard deduction for 2024?",
        "What documents do I need for tax filing?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = qa.ask(q)
        print(f"A: {response['answer']}")
        print(f"Sources: {len(response['sources'])} documents")
