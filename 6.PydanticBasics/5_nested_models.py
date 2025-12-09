"""
Pydantic Basics - Nested Models
Creating models that contain other models
"""

from pydantic import BaseModel

# Simple model
class Address(BaseModel):
    street: str
    city: str
    country: str
    zip_code: str

# Model containing another model
class Person(BaseModel):
    name: str
    age: int
    address: Address  # Nested model

print("=" * 60)
print("NESTED MODELS")
print("=" * 60)

# Create address
address = Address(
    street="123 Main St",
    city="New York",
    country="USA",
    zip_code="10001"
)

# Create person with address
person = Person(
    name="Emma",
    age=28,
    address=address
)

print(f"\n✅ Person with nested Address:")
print(f"  Name: {person.name}")
print(f"  Age: {person.age}")
print(f"  Address:")
print(f"    Street: {person.address.street}")
print(f"    City: {person.address.city}")
print(f"    Country: {person.address.country}")
print(f"    ZIP: {person.address.zip_code}")

# Or create with dictionary (Pydantic converts it)
person2 = Person(
    name="Frank",
    age=35,
    address={
        "street": "456 Oak Ave",
        "city": "Los Angeles",
        "country": "USA",
        "zip_code": "90001"
    }
)

print(f"\n✅ Person created from dictionary:")
print(f"  Name: {person2.name}")
print(f"  City: {person2.address.city}")

# Convert to dictionary
print(f"\n📝 Convert to dictionary:")
person_dict = person.model_dump()
print(f"  {person_dict}")

# Convert to JSON
print(f"\n📝 Convert to JSON:")
person_json = person.model_dump_json(indent=2)
print(f"  {person_json}")

print("\n" + "=" * 60)
