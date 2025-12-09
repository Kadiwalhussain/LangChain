# Pydantic Basics - Tutorial

A basic tutorial for learning **Pydantic**, the data validation library for Python.

## What is Pydantic?

Pydantic is a library that:
- ✅ Validates data using Python type hints
- ✅ Converts data to the correct types automatically
- ✅ Catches errors before they cause problems
- ✅ Makes working with structured data easy

## Why Use Pydantic?

**Without Pydantic:**
```python
user = {"name": "John", "age": "twenty", "email": "john@email"}
# age is a string but should be int - you might not notice until later!
# email format not validated
```

**With Pydantic:**
```python
class User(BaseModel):
    name: str
    age: int
    email: str

user = User(name="John", age="twenty", email="john@email")  # Error caught immediately!
```

## Examples Included

### 1. **What is Pydantic** (`1_what_is_pydantic.py`)
- Basic model definition
- Creating instances
- Understanding validation errors

### 2. **Type Hints and Validation** (`2_type_hints_validation.py`)
- Common types: str, int, float, bool
- Type coercion (automatic conversion)
- How Pydantic validates types

### 3. **Optional Fields** (`3_optional_fields.py`)
- Making fields optional
- Default values
- Required vs optional fields

### 4. **Lists and Collections** (`4_lists_collections.py`)
- Using `List[type]` for lists
- Using `Dict[key, value]` for dictionaries
- Iterating through collections

### 5. **Nested Models** (`5_nested_models.py`)
- Models containing other models
- Accessing nested data
- Converting to JSON

### 6. **Custom Validators** (`6_validators.py`)
- Adding validation logic with `@field_validator`
- Validating specific fields
- Complex validation rules

### 7. **LLM Use Case** (`7_llm_use_case.py`)
- Practical example: structuring LLM responses
- Parsing JSON from LLMs
- Validating LLM output

## How to Use

Run any example:

```bash
python 1_what_is_pydantic.py
python 2_type_hints_validation.py
# ... and so on
```

## Basic Syntax

### Simple Model

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = Person(name="Alice", age=30)
```

### With Optional Fields

```python
from typing import Optional

class Product(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

product = Product(name="Laptop", price=999.99)
```

### With Validation

```python
from pydantic import field_validator

class User(BaseModel):
    age: int
    
    @field_validator('age')
    @classmethod
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v
```

### With Lists

```python
from typing import List

class Team(BaseModel):
    name: str
    members: List[str]

team = Team(name="Python Team", members=["Alice", "Bob"])
```

### With Nested Models

```python
class Address(BaseModel):
    city: str
    country: str

class Person(BaseModel):
    name: str
    address: Address

person = Person(
    name="Alice",
    address=Address(city="NYC", country="USA")
)
```

## Key Benefits

1. **Type Safety** - Catch errors early
2. **Auto Conversion** - `"25"` becomes `25` (if possible)
3. **Documentation** - Your code documents itself
4. **Validation** - Complex rules are simple to write
5. **JSON Support** - Easy conversion to/from JSON
6. **Error Messages** - Clear error messages

## Common Use Cases

✅ **Validating API requests** - Ensure client data is correct
✅ **Structuring LLM responses** - Parse and validate LLM output
✅ **Configuration files** - Validate config data
✅ **Database models** - Structure database data
✅ **Data pipelines** - Validate data at each step

## Next Steps

Once you understand these basics:

- Learn about `Config` class for model configuration
- Explore `Field()` for advanced field options
- Use `root_validator` for multi-field validation
- Combine with FastAPI for web APIs
- Use with LangChain for structured outputs

## Pydantic vs Alternatives

| Feature | Pydantic | dataclasses | attrs |
|---------|----------|-------------|-------|
| Validation | ✅ Built-in | ❌ No | ❌ No |
| Type Hints | ✅ Yes | ✅ Yes | ✅ Yes |
| JSON Support | ✅ Great | ❌ Manual | ❌ Manual |
| Error Messages | ✅ Detailed | ❌ Basic | ❌ Basic |
| Learning Curve | ✅ Easy | ✅ Easy | ⚠️ Medium |

## Installation

```bash
pip install pydantic
```

## Tips

- Always use type hints - Pydantic relies on them
- Use `Optional[type]` when a field can be `None`
- Create validators for complex rules
- Use `model_dump()` to convert to dict
- Use `model_dump_json()` to convert to JSON
- Read error messages - they're very helpful!

---

**Remember:** Pydantic makes your code safer and clearer. Start simple and gradually add validation!
