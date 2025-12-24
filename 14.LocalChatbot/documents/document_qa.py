"""
Document Q&A System
===================

Load documents and answer questions about them using local models.
Supports: PDF, TXT, MD, and more.
"""

import os
from typing import Dict, List, Optional
import logging

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class DocumentQA:
    """Document Q&A system using local models."""
    
    def __init__(self):
        """Initialize document QA system."""
        # Local embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Vector stores for different documents
        self.vectorstores = {}
        
        # LLM
        try:
            self.llm = OllamaLLM(model="mistral", temperature=0.2)
            logger.info("✅ Ollama LLM initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {e}")
            self.llm = None
    
    def _get_loader(self, file_path: str):
        """Get appropriate loader for file type."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif ext in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path)
        else:
            # Try text loader as fallback
            try:
                return TextLoader(file_path, encoding='utf-8')
            except:
                raise ValueError(f"Unsupported file type: {ext}")
    
    def load_document(self, file_path: str, file_id: str) -> Dict:
        """Load and process document."""
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Get loader
            loader = self._get_loader(file_path)
            
            # Load documents
            documents = loader.load()
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No content extracted from document'
                }
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create vector store
            persist_directory = f"./documents_db/{file_id}"
            os.makedirs(persist_directory, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name=f"doc_{file_id}"
            )
            
            # Store for later use
            self.vectorstores[file_id] = vectorstore
            
            return {
                'success': True,
                'chunks_created': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def ask_question(self, question: str, file_id: str) -> Dict:
        """Ask a question about the document."""
        try:
            if not self.llm:
                return {
                    'answer': 'Ollama is not running. Please start it with: ollama serve',
                    'sources': []
                }
            
            # Get vector store
            if file_id not in self.vectorstores:
                # Try to load from disk
                persist_directory = f"./documents_db/{file_id}"
                if os.path.exists(persist_directory):
                    vectorstore = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self.embeddings,
                        collection_name=f"doc_{file_id}"
                    )
                    self.vectorstores[file_id] = vectorstore
                else:
                    return {
                        'answer': 'Document not loaded. Please upload the document first.',
                        'sources': []
                    }
            
            vectorstore = self.vectorstores[file_id]
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Create prompt template
            template = """You are a helpful assistant answering questions about a document.

Use the following pieces of context from the document to answer the question.
If you don't know the answer based on the context, say that you don't know.
Keep your answer concise and based only on the provided context.

Context from document:
{context}

Question: {question}

Answer based on the document:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            # Get answer
            result = qa_chain.invoke({"query": question})
            
            # Extract sources
            sources = []
            for doc in result.get('source_documents', [])[:3]:
                sources.append({
                    'content': doc.page_content[:200] + "...",
                    'metadata': doc.metadata
                })
            
            return {
                'answer': result.get('result', 'No answer generated'),
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f'Error: {str(e)}',
                'sources': []
            }


