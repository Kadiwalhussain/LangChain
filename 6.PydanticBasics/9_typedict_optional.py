"""
TypedDict - Optional Fields
Working with optional fields in TypedDict
"""

from typing import TypedDict, Optional

# Basic TypedDict with all required fields
class Book(TypedDict):
    title: str
    author: str
    year: int

# TypedDict with optional fields (total=False makes all optional)
class BookOptional(TypedDict, total=False):
    title: str
    author: str
    year: int
    isbn: str

# Mixed required and optional
class BookMixed(TypedDict):
    title: str
    author: str
    year: int
    isbn: NotRequired[str]  # Python 3.11+

print("=" * 60)
print("TYPEDICT - OPTIONAL FIELDS")
print("=" * 60)

# Required fields - must include all
book1: Book = {
    "title": "Python 101",
    "author": "John Doe",
    "year": 2023
}

print(f"\n✅ Required fields TypedDict:")
print(f"  {book1}")

# Optional fields - can be partial
book2: BookOptional = {
    "title": "Web Dev",
    "author": "Jane Smith"
    # year and isbn are optional
}

print(f"\n✅ Optional fields TypedDict:")
print(f"  {book2}")

# Full optional
book3: BookOptional = {
    "title": "Data Science",
    "author": "Bob Johnson",
    "year": 2024,
    "isbn": "978-1234567890"
}

print(f"\n✅ Optional with all fields:")
print(f"  {book3}")

print("\n📋 Key Differences:")
print("  total=True (default): All fields required")
print("  total=False: All fields optional")
print("  NotRequired[type]: Specific field is optional")

print("\n" + "=" * 60)
