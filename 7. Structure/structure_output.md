# LangChain Output Parsers - Complete Guide

## Overview

Output Parsers in LangChain are components that structure and transform raw text outputs from Language Models (LLMs) into usable, typed data structures. They help convert unstructured LLM responses into formats like JSON, lists, datetime objects, or custom Pydantic models.

## Why Use Output Parsers?

- **Structured Data**: Convert free-form text into structured formats
- **Type Safety**: Ensure outputs match expected schemas
- **Validation**: Automatically validate LLM outputs
- **Consistency**: Get predictable, parseable responses
- **Integration**: Easier integration with downstream applications

## Core Concepts

Every Output Parser implements two main methods:

1. **`get_format_instructions()`**: Returns a string describing how the LLM should format its output
2. **`parse()`**: Takes the LLM output string and parses it into the desired structure

## Common Output Parser Types

### 1. StrOutputParser

The simplest parser that returns the raw string output.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"topic": "cats"})
print(result)  # Returns: string with the joke
```

### 2. JsonOutputParser

Parses LLM output into JSON/dictionary format.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Extract information about the person.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"query": "John is 30 years old and lives in NYC"})
# Returns: {"name": "John", "age": 30, "city": "NYC"}
```

### 3. PydanticOutputParser

Parses output into Pydantic models with full validation.

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Define your data model
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")

llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="Extract information.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"query": "Alice is 25 and lives in Paris"})
# Returns: Person(name="Alice", age=25, city="Paris")
print(result.name)  # Access typed attributes
```

### 4. CommaSeparatedListOutputParser

Parses comma-separated values into a list.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = CommaSeparatedListOutputParser()
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(
    template="List 5 {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"subject": "colors"})
# Returns: ["red", "blue", "green", "yellow", "purple"]
```

### 5. DatetimeOutputParser

Parses dates and times into Python datetime objects.

```python
from langchain_core.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = DatetimeOutputParser()
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(
    template="When did {event} happen?\n{format_instructions}",
    input_variables=["event"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"event": "the moon landing"})
# Returns: datetime.datetime(1969, 7, 20, ...)
```

### 6. StructuredOutputParser

Creates a parser from a response schema for multiple fields.

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the question")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(
    template="Answer the question.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"question": "What is the capital of France?"})
# Returns: {"answer": "Paris", "source": "General knowledge"}
```

### 7. EnumOutputParser

Restricts output to a predefined set of values.

```python
from langchain_core.output_parsers import EnumOutputParser
from enum import Enum
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

parser = EnumOutputParser(enum=Sentiment)
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(
    template="Classify the sentiment.\n{format_instructions}\nText: {text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"text": "I love this product!"})
# Returns: Sentiment.POSITIVE
```

### 8. XMLOutputParser

Parses XML formatted output.

```python
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = XMLOutputParser()
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(
    template="Format in XML.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"query": "Information about Python"})
# Returns parsed XML as dictionary
```

## Advanced Patterns

### Error Handling with OutputFixingParser

When parsing fails, use OutputFixingParser to automatically fix errors.

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

base_parser = PydanticOutputParser(pydantic_object=Product)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# Even if the output is malformed, fixing_parser will attempt to correct it
malformed_output = '{"name": "Widget" "price": "19.99"}'  # Missing comma
result = fixing_parser.parse(malformed_output)
```

### Retry Parser

Automatically retry with the original prompt if parsing fails.

```python
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

base_parser = PydanticOutputParser(pydantic_object=Product)
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# Will retry if parsing fails
result = retry_parser.parse_with_prompt(output_string, prompt_value)
```

### Custom Output Parser

Create your own custom parser by extending BaseOutputParser.

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class CustomListParser(BaseOutputParser[List[str]]):
    """Parse output with custom delimiter."""
    
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return [item.strip() for item in text.split("|")]
    
    def get_format_instructions(self) -> str:
        return "Please format your response as items separated by | character"

parser = CustomListParser()
result = parser.parse("apple | banana | orange")
# Returns: ["apple", "banana", "orange"]
```

## Best Practices

1. **Always Use Format Instructions**: Include `get_format_instructions()` in your prompts to guide the LLM
2. **Validate Early**: Use Pydantic models for type safety and validation
3. **Handle Errors Gracefully**: Use OutputFixingParser or RetryParser for production
4. **Keep Schemas Simple**: Complex nested structures can confuse LLMs
5. **Test with Real Data**: LLM outputs can be unpredictable; test thoroughly
6. **Use Appropriate Parser**: Match the parser to your data structure needs

## Common Use Cases

- **API Response Formatting**: Structure LLM outputs for API endpoints
- **Database Integration**: Parse outputs into database-ready formats
- **Classification Tasks**: Extract and validate categorical outputs
- **Information Extraction**: Pull structured data from unstructured text
- **Multi-Step Workflows**: Chain parsers in complex pipelines

## Installation

```bash
pip install langchain langchain-core langchain-openai pydantic
```

## Key Takeaways

- Output Parsers bridge the gap between unstructured LLM text and structured application data
- Choose parsers based on your output structure needs (JSON, Pydantic, lists, etc.)
- Always include format instructions in your prompts for best results
- Use error-handling parsers in production for robustness
- Pydantic parsers provide the strongest type safety and validation

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)