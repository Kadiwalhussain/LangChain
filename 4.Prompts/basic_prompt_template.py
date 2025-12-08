from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Basic LLM Prompt Example
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7)

# Simple prompt template
simple_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}"
)

# Chain the prompt with LLM
chain = simple_prompt | llm

result = chain.invoke({"topic": "Python programming"})
print("Simple Prompt Result:")
print(result)
print("\n" + "="*50 + "\n")

# Advanced prompt with multiple variables
advanced_prompt = PromptTemplate(
    input_variables=["language", "skill_level", "topic"],
    template="""You are a {language} tutor for {skill_level} learners.
Explain the concept of {topic} in simple terms with an example.
"""
)

chain2 = advanced_prompt | llm

result2 = chain2.invoke({
    "language": "Python",
    "skill_level": "beginner",
    "topic": "list comprehension"
})
print("Advanced Prompt Result:")
print(result2)
