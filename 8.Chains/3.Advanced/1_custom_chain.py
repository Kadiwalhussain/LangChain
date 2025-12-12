"""
Advanced Chain #1: Custom Chain with Error Handling
Create a custom chain class with built-in error handling
"""

from langchain.chains.base import Chain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import Dict, List
import time

class ValidatedQAChain(Chain):
    """Custom chain that validates and retries questions"""
    
    llm: OllamaLLM
    max_retries: int = 2
    
    @property
    def input_keys(self) -> List[str]:
        return ["question"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["answer", "attempts", "success"]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, any]:
        question = inputs["question"]
        attempts = 0
        
        print(f"\n🔄 Processing question: {question}")
        
        while attempts < self.max_retries:
            attempts += 1
            print(f"\n  Attempt {attempts}/{self.max_retries}...")
            
            try:
                # Generate answer
                prompt = f"""Answer this question clearly and concisely:
Question: {question}

Answer:"""
                
                answer = self.llm.invoke(prompt)
                
                # Validate answer (basic validation)
                if len(answer.strip()) < 10:
                    print(f"  ⚠️  Answer too short, retrying...")
                    time.sleep(1)
                    continue
                
                if "I don't know" in answer or "cannot answer" in answer.lower():
                    print(f"  ⚠️  LLM couldn't answer, retrying...")
                    time.sleep(1)
                    continue
                
                # Success!
                print(f"  ✅ Valid answer received")
                return {
                    "answer": answer,
                    "attempts": attempts,
                    "success": True
                }
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                if attempts >= self.max_retries:
                    return {
                        "answer": "Failed to generate answer",
                        "attempts": attempts,
                        "success": False
                    }
                time.sleep(1)
        
        return {
            "answer": "Max retries exceeded",
            "attempts": attempts,
            "success": False
        }

print("=" * 80)
print("CUSTOM CHAIN WITH ERROR HANDLING")
print("=" * 80)

# Initialize custom chain
llm = OllamaLLM(model="mistral")
custom_chain = ValidatedQAChain(llm=llm, max_retries=3)

# Test questions
questions = [
    "What is the capital of France?",
    "Explain quantum physics in simple terms",
    "What is 2+2?"
]

print("\n🧪 Testing Custom Chain:\n")

for question in questions:
    result = custom_chain.invoke({"question": question})
    print(f"\n📝 Question: {question}")
    print(f"✅ Success: {result['success']}")
    print(f"🔢 Attempts: {result['attempts']}")
    print(f"💬 Answer: {result['answer'][:150]}...")

print("\n" + "=" * 80)
print("CUSTOM CHAIN FEATURES")
print("=" * 80)
print("""
Custom Chain Class:
✅ Inherits from Chain base class
✅ Defines input_keys and output_keys
✅ Implements _call method
✅ Custom validation logic
✅ Retry mechanism
✅ Error handling

Benefits:
- Full control over execution
- Built-in validation
- Automatic retries
- Detailed logging
- Reusable component

Use Cases:
- API calls with retry logic
- Data validation
- Multi-step processes
- Complex business logic
""")
print("=" * 80)
