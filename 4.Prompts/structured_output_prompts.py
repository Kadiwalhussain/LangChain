from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

# Structured Output with Prompts
model = ChatOpenAI(model='gpt-4o', temperature=0.7)

# Define response schema
response_schemas = [
    ResponseSchema(name="title", description="The title of the concept"),
    ResponseSchema(name="definition", description="A clear definition"),
    ResponseSchema(name="example", description="A practical example"),
    ResponseSchema(name="use_cases", description="Common use cases")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Prompt with structured output
prompt = ChatPromptTemplate.from_template("""Explain the following concept in a structured format.

{format_instructions}

Concept: {concept}""")

chain = prompt | model | output_parser

result = chain.invoke({
    "concept": "REST API",
    "format_instructions": format_instructions
})

print("Structured Output Result:")
for key, value in result.items():
    print(f"{key.upper()}: {value}\n")

print("="*50 + "\n")

# Zero-shot Prompting
zero_shot_prompt = ChatPromptTemplate.from_template("""Classify the sentiment of the following text as positive, negative, or neutral.

Text: {text}

Sentiment:""")

chain2 = zero_shot_prompt | model

result2 = chain2.invoke({"text": "I love this product! It works perfectly."})
print("Zero-shot Classification Result:")
print(result2.content)
