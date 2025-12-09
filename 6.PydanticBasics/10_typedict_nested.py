"""
TypedDict - Nested Structures
Creating nested TypedDict structures
"""

from typing import TypedDict, List

# Define nested TypedDicts
class Address(TypedDict):
    street: str
    city: str
    country: str

class Contact(TypedDict):
    email: str
    phone: str

class Employee(TypedDict):
    name: str
    age: int
    address: Address
    contact: Contact
    skills: List[str]

print("=" * 60)
print("TYPEDICT - NESTED STRUCTURES")
print("=" * 60)

# Create nested structure
employee: Employee = {
    "name": "Charlie Brown",
    "age": 32,
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "country": "USA"
    },
    "contact": {
        "email": "charlie@example.com",
        "phone": "555-1234"
    },
    "skills": ["Python", "JavaScript", "SQL", "Docker"]
}

print(f"\n✅ Nested TypedDict structure:")
print(f"  Name: {employee['name']}")
print(f"  Age: {employee['age']}")

print(f"\n📍 Address (nested):")
print(f"  Street: {employee['address']['street']}")
print(f"  City: {employee['address']['city']}")
print(f"  Country: {employee['address']['country']}")

print(f"\n📞 Contact (nested):")
print(f"  Email: {employee['contact']['email']}")
print(f"  Phone: {employee['contact']['phone']}")

print(f"\n🛠️  Skills (list):")
for skill in employee['skills']:
    print(f"  - {skill}")

print(f"\n📊 Full structure:")
import json
print(json.dumps(employee, indent=2))

print("\n" + "=" * 60)
