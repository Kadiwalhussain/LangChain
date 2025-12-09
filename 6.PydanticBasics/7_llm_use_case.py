"""
Pydantic Basics - Practical LLM Use Case
Using Pydantic to structure LLM responses
"""

from pydantic import BaseModel
from typing import List
import json

# Define a structured response model for movie information
class Movie(BaseModel):
    title: str
    year: int
    genre: str
    rating: float
    director: str

class MovieRecommendation(BaseModel):
    query: str
    recommendations: List[Movie]
    total_found: int

print("=" * 60)
print("PYDANTIC FOR LLM STRUCTURED OUTPUT")
print("=" * 60)

# Simulate LLM response as JSON
llm_response_json = """{
    "query": "sci-fi movies from 2023",
    "recommendations": [
        {
            "title": "Oppenheimer",
            "year": 2023,
            "genre": "Biography",
            "rating": 8.5,
            "director": "Christopher Nolan"
        },
        {
            "title": "The Creator",
            "year": 2023,
            "genre": "Science Fiction",
            "rating": 7.2,
            "director": "Gareth Edwards"
        },
        {
            "title": "Killers of the Flower Moon",
            "year": 2023,
            "genre": "Crime Drama",
            "rating": 8.0,
            "director": "Martin Scorsese"
        }
    ],
    "total_found": 3
}"""

print("\n📝 LLM Response (JSON):")
print(llm_response_json)

# Parse and validate with Pydantic
print(f"\n✅ Parsing with Pydantic:")
try:
    data = json.loads(llm_response_json)
    recommendation = MovieRecommendation(**data)
    
    print(f"\n  Query: {recommendation.query}")
    print(f"  Total found: {recommendation.total_found}")
    print(f"\n  Recommendations:")
    for i, movie in enumerate(recommendation.recommendations, 1):
        print(f"    {i}. {movie.title} ({movie.year})")
        print(f"       Director: {movie.director}")
        print(f"       Genre: {movie.genre}")
        print(f"       Rating: {movie.rating}/10")
    
    # Access data programmatically
    print(f"\n📊 Accessing data:")
    best_rated = max(recommendation.recommendations, key=lambda m: m.rating)
    print(f"  Best rated: {best_rated.title} ({best_rated.rating})")
    
    print(f"\n✅ Full validation passed!")
    
except Exception as e:
    print(f"  ❌ Validation error: {e}")

# Example with invalid data
print(f"\n" + "=" * 60)
print("VALIDATION ERROR EXAMPLE")
print("=" * 60)

invalid_response = """{
    "query": "sci-fi movies",
    "recommendations": [
        {
            "title": "Oppenheimer",
            "year": "2023",
            "genre": "Biography",
            "rating": "8.5",
            "director": "Christopher Nolan"
        }
    ],
    "total_found": 1
}"""

print("\n❌ Invalid data (year and rating should be numbers):")
try:
    data = json.loads(invalid_response)
    recommendation = MovieRecommendation(**data)
except Exception as e:
    print(f"  Pydantic caught the error:")
    print(f"  {e}")

print("\n" + "=" * 60)
