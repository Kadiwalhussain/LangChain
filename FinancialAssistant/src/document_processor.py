"""Document processing module for financial documents."""

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os
from pathlib import Path
import logging
from config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process financial documents (PDFs, Word, Excel)."""
    
    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Processing PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_name': Path(file_path).name,
                    'file_type': 'pdf',
                    'category': self._classify_document(file_path)
                })
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from PDF")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def process_docx(self, file_path: str) -> List[Document]:
        """
        Process a Word document.
        
        Args:
            file_path: Path to Word file
            
        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Processing DOCX: {file_path}")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_name': Path(file_path).name,
                    'file_type': 'docx',
                    'category': self._classify_document(file_path)
                })
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from DOCX")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise
    
    def process_excel(self, file_path: str) -> List[Document]:
        """
        Process an Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Processing Excel: {file_path}")
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_name': Path(file_path).name,
                    'file_type': 'xlsx',
                    'category': 'financial_data'
                })
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from Excel")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {e}")
            raise
    
    def process_directory(self, directory: str) -> List[Document]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Combined list of all document chunks
        """
        all_chunks = []
        
        for file_path in Path(directory).rglob("*"):
            if not file_path.is_file():
                continue
                
            try:
                if file_path.suffix.lower() == '.pdf':
                    chunks = self.process_pdf(str(file_path))
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    chunks = self.process_docx(str(file_path))
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    chunks = self.process_excel(str(file_path))
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Processed directory {directory}: {len(all_chunks)} total chunks")
        return all_chunks
    
    def _classify_document(self, file_path: str) -> str:
        """
        Classify document based on filename and content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Document category
        """
        filename = Path(file_path).name.lower()
        
        # Tax forms
        tax_forms = ['1099', 'w2', 'w-2', '1040', 'schedule', 'form']
        if any(form in filename for form in tax_forms):
            return 'tax_form'
        
        # Financial statements
        if any(word in filename for word in ['balance', 'income', 'profit', 'loss', 'statement']):
            return 'financial_statement'
        
        # Receipts/Invoices
        if any(word in filename for word in ['receipt', 'invoice', 'expense']):
            return 'receipt'
        
        # Knowledge base
        if 'guide' in filename or 'regulation' in filename:
            return 'knowledge_base'
        
        return 'general'
    
    def extract_form_fields(self, file_path: str) -> dict:
        """
        Extract structured fields from tax forms.
        
        Args:
            file_path: Path to form PDF
            
        Returns:
            Dictionary of extracted fields
        """
        # This is a simplified version
        # In production, you'd use pdfplumber or form-specific parsing
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            content = " ".join([doc.page_content for doc in documents])
            
            # Extract common fields (simplified)
            fields = {
                'file_name': Path(file_path).name,
                'content_length': len(content),
                'page_count': len(documents)
            }
            
            return fields
            
        except Exception as e:
            logger.error(f"Error extracting form fields: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = DocumentProcessor()
    
    # Example: Process a single PDF
    # chunks = processor.process_pdf("data/tax_forms/sample_form.pdf")
    # print(f"Processed {len(chunks)} chunks")
    
    # Example: Process entire directory
    # all_docs = processor.process_directory("data/knowledge_base")
    # print(f"Total documents: {len(all_docs)}")
