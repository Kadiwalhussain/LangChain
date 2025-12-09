"""
Pydantic Basics - Lists and Collections
Using lists, dicts, and other collections in Pydantic
"""

from pydantic import BaseModel
from typing import List, Dict

# Model with list
class Course(BaseModel):
    name: str
    students: List[str]  # List of strings (student names)
    grades: List[int]    # List of integers

# Model with dictionary
class Library(BaseModel):
    name: str
    books: Dict[str, int]  # Key=book title, Value=quantity

print("=" * 60)
print("LISTS AND COLLECTIONS")
print("=" * 60)

# Create course with list
course = Course(
    name="Python 101",
    students=["Alice", "Bob", "Charlie"],
    grades=[95, 87, 92]
)

print(f"\n✅ Course with Lists:")
print(f"  Name: {course.name}")
print(f"  Students: {course.students}")
print(f"  Grades: {course.grades}")

# Access individual items
print(f"\n📋 Accessing list items:")
print(f"  First student: {course.students[0]}")
print(f"  Second grade: {course.grades[1]}")

# Create library with dictionary
library = Library(
    name="City Library",
    books={
        "Python Guide": 5,
        "Web Development": 3,
        "Data Science": 8
    }
)

print(f"\n✅ Library with Dictionary:")
print(f"  Name: {library.name}")
print(f"  Books: {library.books}")

# Access dictionary items
print(f"\n📋 Accessing dictionary items:")
print(f"  Python Guide count: {library.books['Python Guide']}")

# Iterate through collections
print(f"\n📚 Iterating through students:")
for student in course.students:
    print(f"  - {student}")

print(f"\n📚 Iterating through books:")
for title, count in library.books.items():
    print(f"  - {title}: {count} copies")

print("\n" + "=" * 60)
