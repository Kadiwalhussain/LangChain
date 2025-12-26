"""Tax form analyzer using LLM."""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from config import settings
from src.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class FormAnalysisResult(BaseModel):
    """Form analysis result model."""
    
    form_type: str = Field(description="Type of tax form")
    completeness_score: float = Field(description="Completeness percentage (0-100)")
    missing_fields: List[str] = Field(description="List of missing required fields")
    warnings: List[str] = Field(description="Warnings or issues found")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    summary: str = Field(description="Overall summary of the form")


class FormAnalyzer:
    """Analyze tax forms and financial documents."""
    
    def __init__(self):
        """Initialize form analyzer."""
        self.llm = self._get_llm()
        self.document_processor = DocumentProcessor()
        self.parser = JsonOutputParser(pydantic_object=FormAnalysisResult)
    
    def _get_llm(self):
        """Get LLM instance."""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=0.1,  # Low temperature for consistency
                openai_api_key=settings.openai_api_key
            )
        else:
            return ChatOllama(
                model=settings.ollama_model,
                temperature=0.1,
                base_url=settings.ollama_base_url
            )
    
    def analyze_form(self, file_path: str) -> dict:
        """
        Analyze a tax form for completeness and issues.
        
        Args:
            file_path: Path to form PDF
            
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Analyzing form: {file_path}")
            
            # Load document
            documents = self.document_processor.process_pdf(file_path)
            content = "\n\n".join([doc.page_content for doc in documents])
            
            # Create analysis prompt
            prompt_template = """You are an expert tax accountant analyzing a tax form.

Analyze the following form content and provide a structured analysis.

Form Content:
{content}

Provide your analysis in the following JSON format:
{format_instructions}

Focus on:
1. Identifying the form type (W-2, 1099, 1040, etc.)
2. Checking for missing required fields
3. Identifying potential errors or inconsistencies
4. Providing actionable recommendations

Be thorough but concise."""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["content"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            
            # Create chain
            chain = prompt | self.llm | self.parser
            
            # Analyze
            result = chain.invoke({"content": content[:4000]})  # Limit content length
            
            logger.info("Form analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing form: {e}")
            return {
                "error": str(e),
                "form_type": "Unknown",
                "completeness_score": 0,
                "missing_fields": [],
                "warnings": [f"Analysis failed: {str(e)}"],
                "recommendations": ["Please try again or contact support"],
                "summary": "Unable to complete analysis"
            }
    
    def check_required_fields(self, content: str, form_type: str) -> List[str]:
        """
        Check for required fields based on form type.
        
        Args:
            content: Form content
            form_type: Type of form (W-2, 1099, etc.)
            
        Returns:
            List of missing fields
        """
        # This is a simplified version
        # In production, you'd have comprehensive field definitions per form type
        
        required_fields = {
            "w-2": ["employer_ein", "employee_ssn", "wages", "federal_tax_withheld"],
            "1099": ["payer_ein", "recipient_ssn", "income_amount"],
            "1040": ["ssn", "filing_status", "income", "deductions"]
        }
        
        form_key = form_type.lower().replace("-", "").replace(" ", "")
        fields = required_fields.get(form_key, [])
        
        missing = []
        for field in fields:
            if field.lower() not in content.lower():
                missing.append(field)
        
        return missing
    
    def validate_data_consistency(self, file_path: str) -> dict:
        """
        Validate data consistency within a form.
        
        Args:
            file_path: Path to form
            
        Returns:
            Validation results
        """
        try:
            documents = self.document_processor.process_pdf(file_path)
            content = "\n".join([doc.page_content for doc in documents])
            
            prompt = f"""Review the following tax form for data consistency issues:

{content[:2000]}

Check for:
1. Mathematical errors (totals don't add up)
2. Missing required relationships (e.g., W-2 wages should match 1040 line 1)
3. Invalid values (negative amounts where not allowed)
4. Format issues (invalid SSN, EIN patterns)

List any issues found."""

            result = self.llm.invoke(prompt)
            
            return {
                "file": file_path,
                "validation_result": result.content,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "file": file_path,
                "status": "error",
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    analyzer = FormAnalyzer()
    
    # Example: Analyze a form
    # result = analyzer.analyze_form("data/tax_forms/sample_w2.pdf")
    # print(f"Form Type: {result['form_type']}")
    # print(f"Completeness: {result['completeness_score']}%")
    # print(f"Missing Fields: {result['missing_fields']}")
    # print(f"Summary: {result['summary']}")
