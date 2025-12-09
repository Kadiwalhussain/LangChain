"""
Pydantic Basics - Validators
Adding custom validation logic to fields
"""

from pydantic import BaseModel, field_validator

class User(BaseModel):
    username: str
    email: str
    age: int
    
    @field_validator('username')
    @classmethod
    def username_must_be_alphanumeric(cls, v):
        # Username should only contain letters, numbers, underscores
        if not v.replace('_', '').isalnum():
            raise ValueError('Username must be alphanumeric (and underscores)')
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v
    
    @field_validator('email')
    @classmethod
    def email_must_contain_at(cls, v):
        # Simple email validation
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @field_validator('age')
    @classmethod
    def age_must_be_valid(cls, v):
        # Age should be between 13 and 120
        if v < 13:
            raise ValueError('Must be at least 13 years old')
        if v > 120:
            raise ValueError('Invalid age')
        return v

print("=" * 60)
print("CUSTOM VALIDATORS")
print("=" * 60)

# Valid user
user1 = User(
    username="john_doe",
    email="john@example.com",
    age=25
)

print(f"\n✅ Valid user created:")
print(f"  Username: {user1.username}")
print(f"  Email: {user1.email}")
print(f"  Age: {user1.age}")

# Test validation errors
print(f"\n❌ Testing invalid data:")

test_cases = [
    {"username": "ab", "email": "john@example.com", "age": 25},  # Username too short
    {"username": "john-doe", "email": "john@example.com", "age": 25},  # Username has dash
    {"username": "john_doe", "email": "invalidemail", "age": 25},  # Invalid email
    {"username": "john_doe", "email": "john@example.com", "age": 10},  # Age too young
]

for i, test_data in enumerate(test_cases, 1):
    try:
        user = User(**test_data)
    except Exception as e:
        print(f"\n  Test {i} failed:")
        print(f"    Data: {test_data}")
        print(f"    Error: {e}")

print("\n" + "=" * 60)
