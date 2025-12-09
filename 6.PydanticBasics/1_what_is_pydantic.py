"""
Pydantic Basics - What is Pydantic?
Pydantic is a data validation library that uses Python type hints.
It ensures data is valid and properly typed before you use it.
"""

from pydantic import BaseModel

# Define a simple model
class Person(BaseModel):
    name: str
    age: int
    email: str

# Create an instance with valid data
person = Person(name="Alice", age=30, email="alice@example.com")

print("=" * 60)
print("PYDANTIC BASICS - SIMPLE MODEL")
print("=" * 60)
print(f"\n✅ Created person:")
print(f"  Name: {person.name}")
print(f"  Age: {person.age}")
print(f"  Email: {person.email}")

# Pydantic automatically validates types
print("\n📋 Accessing as dictionary:")
print(f"  {person.model_dump()}")

print("\n📝 JSON representation:")
print(f"  {person.model_dump_json()}")

# What happens with wrong types?
print("\n" + "=" * 60)
print("VALIDATION ERROR EXAMPLE")
print("=" * 60)

try:
    # This will fail - age should be int, not string
    invalid_person = Person(name="Bob", age="thirty", email="bob@example.com")
except Exception as e:
    print(f"\n❌ Validation Error:")
    print(f"  {e}")

print("\n" + "=" * 60)
