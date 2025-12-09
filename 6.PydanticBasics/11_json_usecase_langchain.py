"""
TypedDict + Pydantic - JSON Use Case in LangChain
Using TypedDict and Pydantic together for LLM JSON responses
"""

from typing import TypedDict, List
from pydantic import BaseModel, field_validator
import json

# Define the structure with TypedDict (for documentation)
class MovieDict(TypedDict):
    title: str
    year: int
    genre: str
    rating: float
    director: str

class MovieRecommendationDict(TypedDict):
    query: str
    movies: List[MovieDict]
    total_count: int

# Define Pydantic models for validation
class Movie(BaseModel):
    title: str
    year: int
    genre: str
    rating: float
    director: str
    
    @field_validator('year')
    @classmethod
    def year_must_be_valid(cls, v):
        if v < 1800 or v > 2100:
            raise ValueError('Invalid year')
        return v
    
    @field_validator('rating')
    @classmethod
    def rating_must_be_between_0_and_10(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Rating must be between 0 and 10')
        return v

class MovieRecommendation(BaseModel):
    query: str
    movies: List[Movie]
    total_count: int

print("=" * 60)
print("TYPEDICT + PYDANTIC - JSON USE CASE IN LANGCHAIN")
print("=" * 60)

# Simulate LLM response (JSON)
llm_json_response = """{
    "query": "best sci-fi movies",
    "movies": [
        {
            "title": "Inception",
            "year": 2010,
            "genre": "Science Fiction",
            "rating": 8.8,
            "director": "Christopher Nolan"
        },
        {
            "title": "Interstellar",
            "year": 2014,
            "genre": "Science Fiction",
            "rating": 8.6,
            "director": "Christopher Nolan"
        },
        {
            "title": "The Matrix",
            "year": 1999,
            "genre": "Science Fiction",
            "rating": 8.7,
            "director": "Lana Wachowski, Lilly Wachowski"
        }
    ],
    "total_count": 3
}"""

print(f"\n📝 LLM JSON Response:")
print(llm_json_response)

# Parse and validate with Pydantic
print(f"\n{'=' * 60}")
print(f"PARSING WITH PYDANTIC")
print(f"{'=' * 60}")

try:
    data = json.loads(llm_json_response)
    recommendation = MovieRecommendation(**data)
    
    print(f"\n✅ Successfully validated!")
    print(f"\n📊 Recommendation Details:")
    print(f"  Query: {recommendation.query}")
    print(f"  Total movies: {recommendation.total_count}")
    
    print(f"\n🎬 Movies:")
    for i, movie in enumerate(recommendation.movies, 1):
        print(f"\n  {i}. {movie.title}")
        print(f"     Year: {movie.year}")
        print(f"     Genre: {movie.genre}")
        print(f"     Rating: {movie.rating}/10")
        print(f"     Director: {movie.director}")
    
    # Use the data programmatically
    print(f"\n{'=' * 60}")
    print(f"PROGRAMMATIC ACCESS")
    print(f"{'=' * 60}")
    
    # Find highest rated
    highest_rated = max(recommendation.movies, key=lambda m: m.rating)
    print(f"\n⭐ Highest rated: {highest_rated.title} ({highest_rated.rating}/10)")
    
    # Filter by year
    recent_movies = [m for m in recommendation.movies if m.year >= 2010]
    print(f"\n📅 Movies from 2010 onwards:")
    for movie in recent_movies:
        print(f"   - {movie.title} ({movie.year})")
    
    # Convert back to dict for API response
    print(f"\n{'=' * 60}")
    print(f"CONVERT BACK TO JSON FOR API RESPONSE")
    print(f"{'=' * 60}")
    
    response_json = recommendation.model_dump_json(indent=2)
    print(f"\n{response_json}")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing error: {e}")
except Exception as e:
    print(f"❌ Validation error: {e}")

# Example with invalid data
print(f"\n{'=' * 60}")
print(f"VALIDATION ERROR - INVALID DATA")
print(f"{'=' * 60}")

invalid_response = """{
    "query": "movies",
    "movies": [
        {
            "title": "Movie",
            "year": 3000,
            "genre": "Drama",
            "rating": 15,
            "director": "Someone"
        }
    ],
    "total_count": 1
}"""

print(f"\n❌ Invalid data (year > 2100, rating > 10):")
try:
    data = json.loads(invalid_response)
    recommendation = MovieRecommendation(**data)
except Exception as e:
    print(f"  Pydantic caught the error:")
    print(f"  {e}")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print(f"""
1. TypedDict: Define structure (no validation at runtime)
2. Pydantic: Validate + convert data at runtime
3. Together: Perfect for LLM responses!

Why this approach?
✅ LLM returns JSON
✅ Pydantic validates it immediately
✅ Type hints show expected structure
✅ Custom validators ensure data quality
✅ Easy to convert back to JSON for API responses
✅ Type-safe access to all fields
""")
print("=" * 60)
