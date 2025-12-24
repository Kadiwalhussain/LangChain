"""
General Chat Module
==================

Free-form conversation using local Ollama LLM.
No API keys required!
"""

import logging
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class GeneralChat:
    """General conversation chatbot."""
    
    def __init__(self):
        """Initialize general chat."""
        try:
            self.llm = OllamaLLM(model="mistral", temperature=0.7)
            logger.info("✅ Ollama LLM initialized for general chat")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {e}")
            self.llm = None
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            if self.llm:
                # Try a simple test
                test_response = self.llm.invoke("Hello")
                return True
            return False
        except:
            return False
    
    def chat(self, message: str, history: List[Dict] = None) -> str:
        """Get chat response."""
        try:
            if not self.llm:
                return "Ollama is not running. Please start it with: ollama serve"
            
            # Build conversation context
            messages = []
            
            # Add history if provided
            if history:
                for item in history[-5:]:  # Last 5 messages for context
                    if item.get('role') == 'user':
                        messages.append(HumanMessage(content=item.get('content', '')))
                    elif item.get('role') == 'assistant':
                        messages.append(AIMessage(content=item.get('content', '')))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Get response
            response = self.llm.invoke(messages)
            
            return response if isinstance(response, str) else response.content
            
        except Exception as e:
            logger.error(f"Error in general chat: {str(e)}")
            return f"Error: {str(e)}. Make sure Ollama is running: ollama serve"


