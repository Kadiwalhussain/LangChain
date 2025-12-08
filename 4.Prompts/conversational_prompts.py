from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Conversational Prompting
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7)

# Multi-turn conversation prompt
conversation_prompt = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""You are a helpful AI assistant.

Context: {context}

User: {user_input}
Assistant:"""
)

chain = conversation_prompt | llm

# Simulate conversation
context = "The user is asking about web development"
response1 = chain.invoke({
    "context": context,
    "user_input": "What is the difference between frontend and backend?"
})

print("Response 1:")
print(response1)
print("\n" + "="*50 + "\n")

# Prompt Chaining
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following in 2-3 sentences:\n{text}"
)

chain2 = summary_prompt | llm

result = chain2.invoke({"text": response1})
print("Summarized Response:")
print(result)
