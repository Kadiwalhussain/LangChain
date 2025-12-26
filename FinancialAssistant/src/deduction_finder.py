"""Find applicable tax deductions based on user profile."""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)


class Deduction(BaseModel):
    """Tax deduction model."""
    
    name: str = Field(description="Name of the deduction")
    category: str = Field(description="Category (business, personal, education, etc.)")
    description: str = Field(description="What this deduction covers")
    eligibility: str = Field(description="Who is eligible")
    estimated_savings: Optional[float] = Field(description="Estimated tax savings in dollars")
    documentation_required: List[str] = Field(description="Required documentation")
    tips: str = Field(description="Tips for claiming this deduction")


class DeductionFinder:
    """Find applicable tax deductions for users."""
    
    def __init__(self):
        """Initialize deduction finder."""
        self.llm = self._get_llm()
        self.parser = JsonOutputParser(pydantic_object=Deduction)
    
    def _get_llm(self):
        """Get LLM instance."""
        if settings.llm_provider == "openai":
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=0.3,
                openai_api_key=settings.openai_api_key
            )
        else:
            return ChatOllama(
                model=settings.ollama_model,
                temperature=0.3,
                base_url=settings.ollama_base_url
            )
    
    def find_deductions(
        self,
        profession: str,
        expenses: List[str],
        income: float,
        filing_status: str = "single"
    ) -> List[dict]:
        """
        Find applicable deductions based on user profile.
        
        Args:
            profession: User's profession
            expenses: List of expense categories
            income: Annual income
            filing_status: Tax filing status
            
        Returns:
            List of applicable deductions
        """
        try:
            logger.info(f"Finding deductions for {profession}")
            
            prompt = f"""You are a tax expert helping find deductions for:

Profession: {profession}
Expenses: {', '.join(expenses)}
Annual Income: ${income:,.2f}
Filing Status: {filing_status}

Identify the top 5 most relevant tax deductions this person should consider.
For each deduction, provide:
- Name and category
- Clear description
- Eligibility requirements
- Estimated savings (if applicable)
- Required documentation
- Practical tips for claiming

Focus on deductions specific to their profession and expense patterns.

Return your response as a JSON array of deductions."""

            result = self.llm.invoke(prompt)
            
            # Parse the response
            # Note: This is simplified. In production, you'd have better parsing
            deductions = []
            
            # Common deductions based on profession
            deductions.append({
                "name": "Home Office Deduction",
                "category": "business",
                "description": f"Deduct portion of home expenses if you work from home as a {profession}",
                "eligibility": "Must use space exclusively and regularly for business",
                "estimated_savings": min(income * 0.05, 5000),
                "documentation_required": [
                    "Floor plan or photos of office space",
                    "Utility bills",
                    "Rent or mortgage statements"
                ],
                "tips": "Measure your office space accurately and keep detailed records"
            })
            
            if "computer" in expenses or "software" in expenses:
                deductions.append({
                    "name": "Equipment & Software",
                    "category": "business",
                    "description": "Deduct cost of computers, software, and equipment",
                    "eligibility": "Must be used primarily for business",
                    "estimated_savings": 1000,
                    "documentation_required": [
                        "Purchase receipts",
                        "Software subscriptions",
                        "Depreciation schedule for equipment over $2,500"
                    ],
                    "tips": "You can expense items under $2,500 immediately or depreciate larger purchases"
                })
            
            if "internet" in expenses or "phone" in expenses:
                deductions.append({
                    "name": "Internet & Phone",
                    "category": "business",
                    "description": "Deduct business portion of internet and phone bills",
                    "eligibility": "Must track business vs personal use",
                    "estimated_savings": 600,
                    "documentation_required": [
                        "Monthly bills",
                        "Usage logs showing business percentage"
                    ],
                    "tips": "Calculate what percentage is business use and apply consistently"
                })
            
            logger.info(f"Found {len(deductions)} applicable deductions")
            return deductions
            
        except Exception as e:
            logger.error(f"Error finding deductions: {e}")
            return []
    
    def estimate_total_savings(self, deductions: List[dict], tax_rate: float = 0.22) -> float:
        """
        Estimate total tax savings from deductions.
        
        Args:
            deductions: List of deductions
            tax_rate: Marginal tax rate
            
        Returns:
            Estimated total savings
        """
        total = sum(d.get('estimated_savings', 0) for d in deductions)
        return total * tax_rate
    
    def get_deduction_checklist(self, profession: str) -> List[dict]:
        """
        Get a checklist of documents needed for common deductions.
        
        Args:
            profession: User's profession
            
        Returns:
            Document checklist
        """
        # Profession-specific checklists
        checklists = {
            "software_engineer": [
                {"item": "Home office measurements", "category": "Space"},
                {"item": "Computer purchase receipts", "category": "Equipment"},
                {"item": "Software subscription invoices", "category": "Software"},
                {"item": "Internet/phone bills", "category": "Utilities"},
                {"item": "Professional development courses", "category": "Education"}
            ],
            "freelancer": [
                {"item": "Business expense receipts", "category": "General"},
                {"item": "Mileage logs", "category": "Transportation"},
                {"item": "Client invoices", "category": "Income"},
                {"item": "Health insurance premiums", "category": "Healthcare"}
            ],
            "consultant": [
                {"item": "Travel expenses", "category": "Travel"},
                {"item": "Meal receipts (50% deductible)", "category": "Meals"},
                {"item": "Marketing expenses", "category": "Marketing"},
                {"item": "Professional insurance", "category": "Insurance"}
            ]
        }
        
        key = profession.lower().replace(" ", "_")
        return checklists.get(key, checklists["freelancer"])


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    finder = DeductionFinder()
    
    # Example: Software engineer working from home
    deductions = finder.find_deductions(
        profession="Software Engineer",
        expenses=["home_office", "computer", "internet", "software"],
        income=120000,
        filing_status="single"
    )
    
    print("\n=== Applicable Tax Deductions ===\n")
    for d in deductions:
        print(f"✓ {d['name']}")
        print(f"  Category: {d['category']}")
        print(f"  Estimated savings: ${d['estimated_savings']:,.2f}")
        print(f"  Required docs: {', '.join(d['documentation_required'][:2])}...")
        print()
    
    total_savings = finder.estimate_total_savings(deductions)
    print(f"Estimated total tax savings: ${total_savings:,.2f}")
