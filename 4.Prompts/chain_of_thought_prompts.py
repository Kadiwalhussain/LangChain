from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Chain of Thought Prompting
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7)

# Chain of Thought template
cot_prompt = PromptTemplate(
    input_variables=["problem"],
    template="""Solve the following problem step by step:

Problem: {problem}

Let's think about this step by step:
1. First, identify what we know
2. Then, break down the problem
3. Finally, solve it

Solution:"""
)

chain = cot_prompt | llm

result = chain.invoke({
    "problem": "If a train travels 100 km in 2 hours, what is its average speed?"
})

print("Chain of Thought Result:")
print(result)
print("\n" + "="*50 + "\n")

# Role-based Prompting
role_prompt = PromptTemplate(
    input_variables=["role", "task"],
    template="""You are a {role}.
Your task: {task}

Provide a detailed response:"""
)

chain2 = role_prompt | llm

result2 = chain2.invoke({
    "role": "senior Python developer",
    "task": "explain why list comprehension is preferred over for loops"
})

print("Role-based Prompt Result:")
print(result2)
