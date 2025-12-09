"""
TypedDict Basics - What is TypedDict?
TypedDict is a way to define the structure of dictionaries with type hints.
Unlike Pydantic, TypedDict doesn't validate at runtime - it's mainly for type checking.
"""

from typing import TypedDict

# Define a TypedDict
class Person(TypedDict):
    name: str
    age: int
    email: str

# Create a dictionary that matches the TypedDict structure
person: Person = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
}

print("=" * 60)
print("TYPEDICT BASICS")
print("=" * 60)

print(f"\n✅ Created person with TypedDict:")
print(f"  Name: {person['name']}")
print(f"  Age: {person['age']}")
print(f"  Email: {person['email']}")

print(f"\n📋 Full dictionary:")
print(f"  {person}")

# Key difference from Pydantic - TypedDict doesn't validate at runtime
print(f"\n⚠️  TypedDict vs Pydantic:")
print(f"  TypedDict: No runtime validation (type checker only)")
print(f"  Pydantic: Runtime validation + conversion")

# This would fail type checking but runs without error
wrong_person: Person = {
    "name": "Bob",
    "age": "thirty",  # Should be int - but TypedDict won't complain at runtime!
    "email": "bob@example.com"
}

print(f"\n❌ TypedDict doesn't catch this error at runtime:")
print(f"  {wrong_person}")
print(f"  Age type: {type(wrong_person['age']).__name__} (should be int)")

print("\n" + "=" * 60)
