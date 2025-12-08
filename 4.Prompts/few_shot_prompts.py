from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Chat Model with Prompt Engineering
model = ChatOpenAI(model='gpt-4o', temperature=0.7)

# System message template
system_template = "You are a helpful assistant that explains {topic} in a {style} manner."
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human message template
human_template = "{question}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Combine prompts
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Chain with model
chain = chat_prompt | model

result = chain.invoke({
    "topic": "Machine Learning",
    "style": "simple and beginner-friendly",
    "question": "What is a neural network?"
})

print("Chat Prompt Result:")
print(result.content)
print("\n" + "="*50 + "\n")

# Few-shot examples with chat
examples = [
    {
        "input": "What is 2 + 2?",
        "output": "4. Basic arithmetic: 2 + 2 = 4"
    },
    {
        "input": "What is 5 * 3?",
        "output": "15. Basic arithmetic: 5 * 3 = 15"
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Answer the following question:\n{input}",
    input_variables=["input"]
)

chain2 = few_shot_prompt | model

result2 = chain2.invoke({"input": "What is 10 + 5?"})
print("Few-shot Chat Result:")
print(result2.content)
