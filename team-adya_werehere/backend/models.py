from enum import Enum

from pydantic import BaseModel, Field


class HygieneLevel(str, Enum):
    VERY_GOOD = "very good"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    BAD = "bad"
    VERY_BAD = "very bad"
    NEEDS_IMPROVEMENT = "needs improvement"


class Coordinates(BaseModel):
    lat: float | None = None
    lng: float | None = None


class Contact(BaseModel):
    phone: str | None = ""
    email: str | None = ""
    website: str | None = None


class DietaryOptions(BaseModel):
    vegetarian: bool | None = False
    jain: bool | None = False
    vegan: bool | None = False
    halal: bool | None = False
    kosher: bool | None = False
    gluten_free: bool | None = False


class Restaurant(BaseModel):
    name: str
    location: str | None = ""
    contact: Contact
    cuisines: list[str]
    dietary_options: DietaryOptions
    certification: str | None = ""
    rating: float | None = 0.0
    quality: float | None = 0.0
    hygiene: HygieneLevel
    coords: Coordinates | None = None


class SearchRequest(BaseModel):
    query: str | None = None
    cuisine_types: list[str] | None = None
    min_rating: float | None = Field(None, ge=0, le=5)
    min_quality: float | None = Field(None, ge=0, le=10)
    min_hygiene: HygieneLevel | None = None
    dietary_preferences: DietaryOptions | None = None
    location: str | None = None
    page: int = Field(default=1, gt=0)
    page_size: int = Field(default=10, gt=0, le=100)


class SearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[Restaurant]
