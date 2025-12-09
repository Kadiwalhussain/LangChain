"""
Pydantic Basics - Optional Fields
Making fields optional or required with defaults
"""

from pydantic import BaseModel
from typing import Optional

# Without Optional - all fields required
class Product(BaseModel):
    name: str
    price: float
    description: str

# Try creating without all fields - will fail
try:
    product = Product(name="Laptop", price=999.99)
except Exception as e:
    print("❌ Missing field error (description required):")
    print(f"  {e}\n")

# With Optional - fields can be None
class ProductWithOptional(BaseModel):
    name: str
    price: float
    description: Optional[str] = None  # Optional with default None
    discount: Optional[float] = 0.0    # Optional with default 0.0

print("=" * 60)
print("OPTIONAL FIELDS")
print("=" * 60)

# Create with minimum required fields
product1 = ProductWithOptional(name="Laptop", price=999.99)

print(f"\n✅ Created with only required fields:")
print(f"  Name: {product1.name}")
print(f"  Price: {product1.price}")
print(f"  Description: {product1.description} (None)")
print(f"  Discount: {product1.discount} (default 0.0)")

# Create with all fields
product2 = ProductWithOptional(
    name="Mouse",
    price=29.99,
    description="Wireless mouse",
    discount=0.1
)

print(f"\n✅ Created with all fields:")
print(f"  Name: {product2.name}")
print(f"  Price: {product2.price}")
print(f"  Description: {product2.description}")
print(f"  Discount: {product2.discount}")

print("\n" + "=" * 60)
