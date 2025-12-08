from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()

# Output Parsing with Prompts
model = ChatOpenAI(model='gpt-4o', temperature=0.7)

output_parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template("""List {number} {item_type} related to {topic}.
{format_instructions}""")

chain = prompt | model | output_parser

result = chain.invoke({
    "number": 5,
    "item_type": "Python libraries",
    "topic": "Machine Learning",
    "format_instructions": output_parser.get_format_instructions()
})

print("Output Parsed Result:")
for i, item in enumerate(result, 1):
    print(f"{i}. {item}")

print("\n" + "="*50 + "\n")

# Creative Writing Prompt
creative_prompt = ChatPromptTemplate.from_template("""Write a {length} story about {theme}.
Include: {elements}""")

chain2 = creative_prompt | model

result2 = chain2.invoke({
    "length": "short",
    "theme": "AI assistant helping humans",
    "elements": "conflict, resolution, and a moral lesson"
})

print("Creative Writing Result:")
print(result2.content)
