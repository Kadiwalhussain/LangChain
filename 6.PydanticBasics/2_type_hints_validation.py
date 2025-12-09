"""
Pydantic Basics - Type Hints and Validation
Understanding how type hints work in Pydantic
"""

from pydantic import BaseModel

# Simple types
class Student(BaseModel):
    name: str          # Must be text
    age: int           # Must be integer
    gpa: float         # Must be decimal number
    is_active: bool    # Must be True or False

# Test with correct types
student = Student(
    name="Charlie",
    age=20,
    gpa=3.85,
    is_active=True
)

print("=" * 60)
print("TYPE HINTS AND VALIDATION")
print("=" * 60)

print(f"\n✅ Valid data:")
print(f"  Name (str): {student.name}")
print(f"  Age (int): {student.age}")
print(f"  GPA (float): {student.gpa}")
print(f"  Active (bool): {student.is_active}")

# Pydantic tries to convert compatible types
print(f"\n🔄 Type coercion (Pydantic converts if possible):")

student2 = Student(
    name="Diana",
    age="21",           # String that looks like number
    gpa="3.9",          # String that looks like float
    is_active=1         # 1 becomes True
)

print(f"  Name: {student2.name} (type: {type(student2.name).__name__})")
print(f"  Age: {student2.age} (type: {type(student2.age).__name__})")
print(f"  GPA: {student2.gpa} (type: {type(student2.gpa).__name__})")
print(f"  Active: {student2.is_active} (type: {type(student2.is_active).__name__})")

# What types are available?
print(f"\n📚 Available Types:")
print(f"  str - Text")
print(f"  int - Whole numbers")
print(f"  float - Decimal numbers")
print(f"  bool - True/False")
print(f"  list - Multiple items")
print(f"  dict - Key-value pairs")

print("\n" + "=" * 60)
