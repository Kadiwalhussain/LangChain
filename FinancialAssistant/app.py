"""FastAPI application for Financial Assistant."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import shutil
from pathlib import Path
import logging

from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager
from src.qa_chain import FinancialQAChain
from src.form_analyzer import FormAnalyzer
from src.deduction_finder import DeductionFinder
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Financial Assistant API",
    description="LangChain-powered Financial & Tax Assistant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy loading)
_qa_chain: Optional[FinancialQAChain] = None
_vector_manager: Optional[VectorStoreManager] = None
_form_analyzer: Optional[FormAnalyzer] = None
_deduction_finder: Optional[DeductionFinder] = None


def get_qa_chain() -> FinancialQAChain:
    """Get or create QA chain instance."""
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = FinancialQAChain()
    return _qa_chain


def get_vector_manager() -> VectorStoreManager:
    """Get or create vector manager instance."""
    global _vector_manager
    if _vector_manager is None:
        _vector_manager = VectorStoreManager()
        _vector_manager.initialize_store()
    return _vector_manager


def get_form_analyzer() -> FormAnalyzer:
    """Get or create form analyzer instance."""
    global _form_analyzer
    if _form_analyzer is None:
        _form_analyzer = FormAnalyzer()
    return _form_analyzer


def get_deduction_finder() -> DeductionFinder:
    """Get or create deduction finder instance."""
    global _deduction_finder
    if _deduction_finder is None:
        _deduction_finder = DeductionFinder()
    return _deduction_finder


# Pydantic models
class QuestionRequest(BaseModel):
    """Question request model."""
    question: str
    category: Optional[str] = None


class DeductionRequest(BaseModel):
    """Deduction finder request model."""
    profession: str
    expenses: List[str]
    income: float
    filing_status: str = "single"


# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "upload": "/upload",
            "analyze": "/analyze/{filename}",
            "deductions": "/deductions",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        vector_manager = get_vector_manager()
        stats = vector_manager.get_collection_stats()
        
        return {
            "status": "healthy",
            "vector_store": "connected",
            "total_documents": stats.get('total_documents', 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about financial documents.
    
    - **question**: The question to ask
    - **category**: Optional document category filter
    """
    try:
        qa_chain = get_qa_chain()
        
        if request.category:
            result = qa_chain.ask_with_filter(request.question, request.category)
        else:
            result = qa_chain.ask(request.question)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and index a financial document.
    
    Supports PDF, Word, and Excel files.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB"
        )
    
    try:
        # Save file
        upload_path = Path(settings.upload_dir) / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and index in background
        def process_and_index():
            processor = DocumentProcessor()
            vector_manager = get_vector_manager()
            
            if file_ext == '.pdf':
                docs = processor.process_pdf(str(upload_path))
            elif file_ext in ['.docx', '.doc']:
                docs = processor.process_docx(str(upload_path))
            else:
                docs = processor.process_excel(str(upload_path))
            
            vector_manager.add_documents(docs)
            logger.info(f"Indexed {len(docs)} chunks from {file.filename}")
        
        if background_tasks:
            background_tasks.add_task(process_and_index)
        else:
            process_and_index()
        
        return {
            "message": "File uploaded and indexed successfully",
            "filename": file.filename,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{filename}")
async def analyze_form_endpoint(filename: str):
    """
    Analyze a tax form for completeness and issues.
    
    - **filename**: Name of the uploaded file to analyze
    """
    file_path = Path(settings.upload_dir) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        analyzer = get_form_analyzer()
        result = analyzer.analyze_form(str(file_path))
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing form: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deductions")
async def find_deductions_endpoint(request: DeductionRequest):
    """
    Find applicable tax deductions.
    
    - **profession**: User's profession
    - **expenses**: List of expense categories
    - **income**: Annual income
    - **filing_status**: Tax filing status
    """
    try:
        finder = get_deduction_finder()
        deductions = finder.find_deductions(
            profession=request.profession,
            expenses=request.expenses,
            income=request.income,
            filing_status=request.filing_status
        )
        
        total_savings = finder.estimate_total_savings(deductions)
        
        return {
            "deductions": deductions,
            "total_estimated_savings": round(total_savings, 2),
            "count": len(deductions)
        }
    
    except Exception as e:
        logger.error(f"Error finding deductions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    try:
        vector_manager = get_vector_manager()
        stats = vector_manager.get_collection_stats()
        
        return {
            "total_documents": stats.get('total_documents', 0),
            "collection_name": stats.get('collection_name'),
            "embedding_model": stats.get('embedding_model'),
            "vector_store_path": settings.vector_store_path,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from vector store (use with caution)."""
    try:
        vector_manager = get_vector_manager()
        vector_manager.delete_collection()
        
        # Reinitialize
        vector_manager.initialize_store()
        
        return {"message": "All documents cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
