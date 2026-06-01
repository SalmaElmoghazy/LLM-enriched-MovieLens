from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
# Alternatively, if using pydantic natively: from pydantic import BaseModel, Field

class MovieProfile(BaseModel):
    plot_summary: str = Field(description="Concise summary of the plot (max 2 sentences)")
    main_themes: List[str] = Field(description="Key themes (e.g., friendship, betrayal)")
    mood_atmosphere: List[str] = Field(description="Adjectives describing mood")
    target_audience: str = Field(description="Intended demographic")
    visual_style: str = Field(description="Descriptive sentence of visual characteristics")
    cultural_and_Language_context: List[str] = Field(description="Language and cultural setting")

class UserProfile(BaseModel):
    user_id: int
    preference_summary: str = Field(description="Sentence summarizing user taste")
    favorite_genres: List[str] = Field(description="The user's top 3-5 most watched and highly rated genres, in order of preference")
    thematic_interests: List[str] = Field(description="Preferred themes derived from history")
    visual_style_preferences: List[str] = Field(description="Preferred visual styles")
