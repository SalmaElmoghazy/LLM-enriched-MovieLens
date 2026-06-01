import re
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.schemas import MovieProfile, UserProfile

def clean_json_string(text) -> str:
    # Handle both raw strings and LangChain AIMessage objects
    raw_string = text.content if hasattr(text, 'content') else str(text)
    
    # Defensive Regex 1: Un-escape single quotes (e.g., \' or \\' becomes ')
    cleaned = re.sub(r'\\+\'', "'", raw_string)
    
    # Defensive Regex 2: Un-escape ampersands (e.g., \& or \\& becomes &)
    cleaned = re.sub(r'\\+&', "&", cleaned)
    
    return cleaned

regex_cleaner = RunnableLambda(clean_json_string)

def build_chains(llm):
    # --- Movie Parser ---
    movie_parser = JsonOutputParser(pydantic_object=MovieProfile)
    
    movie_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert movie data analyst. Your goal is to output a strictly formatted JSON object.
Use your internal knowledge base to fill in plot, themes, and style.
CRITICAL: 
1. DO NOT use any double quotes (" ") inside your text summaries or string values. 
2. Use single quotes (' ') instead.
3. DO NOT escape single quotes (write don't, NOT don\\'t).
Ensure the output matches the requested JSON schema exactly. Output strictly valid JSON. Do not output any preamble.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a semantic movie profile for:
Title: {title}
Genres: {genres}

{format_instructions}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    movie_prompt = PromptTemplate(
        template=movie_template,
        input_variables=["title", "genres"],
        partial_variables={"format_instructions": movie_parser.get_format_instructions()}
    )
    
    movie_chain = movie_prompt | llm | regex_cleaner | movie_parser
    
    # --- User Parser ---
    user_parser = JsonOutputParser(pydantic_object=UserProfile)
    
    user_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert user profiler. Analyze the user's highest-rated movies to infer their preferences.
CRITICAL: 
1. DO NOT use any double quotes (" ") inside your text summaries or string values. 
2. Use single quotes (' ') instead.
3. DO NOT escape single quotes (write don't, NOT don\\'t).
Ensure the output matches the requested JSON schema exactly. Output strictly valid JSON. Do not output any preamble.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze this user based on their top rated movies:
User ID: {user_id}
User History (Top Rated):
{user_history}

Instructions:
1. Synthesize commonalities (themes, moods).
2. Identify the user's top 3-5 favorite genres based on this history, ordered by preference.

{format_instructions}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    user_prompt = PromptTemplate(
        template=user_template,
        input_variables=["user_id", "user_history"],
        partial_variables={"format_instructions": user_parser.get_format_instructions()}
    )
    
    user_chain = user_prompt | llm | regex_cleaner | user_parser
    
    return movie_chain, user_chain
