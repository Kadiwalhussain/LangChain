from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Translation Prompts
model = ChatOpenAI(model='gpt-4o', temperature=0.7)

translation_prompt = ChatPromptTemplate.from_template("""Translate the following text to {target_language}.
Maintain the tone and style of the original text.

Text: {text}

Translation:""")

chain = translation_prompt | model

result = chain.invoke({
    "target_language": "Spanish",
    "text": "Machine learning is transforming the technology industry"
})

print("Translation Result:")
print(result.content)
print("\n" + "="*50 + "\n")

# Code Generation Prompts
code_prompt = ChatPromptTemplate.from_template("""Write a {language} function to {task}.
Include docstring and error handling.

Requirements: {requirements}""")

chain2 = code_prompt | model

result2 = chain2.invoke({
    "language": "Python",
    "task": "reverse a string",
    "requirements": "handle empty strings, use efficient algorithm"
})

print("Code Generation Result:")
print(result2.content)
print("\n" + "="*50 + "\n")

# Summarization Prompt
summarization_prompt = ChatPromptTemplate.from_template("""Summarize the following text in {format}.
Focus on: {focus_points}

Text: {text}

Summary:""")

chain3 = summarization_prompt | model

result3 = chain3.invoke({
    "format": "bullet points",
    "focus_points": "key concepts, practical applications, limitations",
    "text": """Artificial Intelligence has revolutionized numerous industries.
    Machine learning enables computers to learn from data.
    Deep learning uses neural networks for complex pattern recognition.
    Natural language processing helps machines understand human language.
    These technologies have applications in healthcare, finance, and education."""
})

print("Summarization Result:")
print(result3.content)
