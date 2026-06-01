# LLM-Enhanced MovieLens Profile Generation

This project enriches the user and item metadata of the MovieLens 100k and 1m datasets using the Llama 3.1 8B Instruct LLM. 


## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Authentication:**
   Set your HuggingFace token as an environment variable to download the Llama model:
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="your_huggingface_token_here"
   
   # Linux / macOS
   export HF_TOKEN="your_huggingface_token_here"
   ```

## Usage

Run the pipeline by specifying the dataset (`100k` or `1m`):

```bash
# For MovieLens 100k
python -m src.main --dataset 100k

# For MovieLens 1m
python -m src.main --dataset 1m

# To specify a custom data directory
python -m src.main --dataset 100k --data-dir custom_data_folder
```

**Note on datasets:** The script expects the original MovieLens datasets to exist in the configured data directory (e.g., `data/ml-100k` or `data/ml-1m` by default). If the directories do not exist, the script will automatically create the data directory and download them from the official GroupLens URLs:
- [MovieLens 100k](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
- [MovieLens 1m](https://files.grouplens.org/datasets/movielens/ml-1m.zip)

The script will automatically:
1. Download and extract the dataset (if missing).
2. Process cold-start splits (Random movies isolated to simulate cold-start items, hence didn't contribute to building user profiles).
3. Generate JSON profiles for movies and users.
4. Checkpoint progress automatically (safe to interrupt and resume).

Results will be saved in the `results/` directory.

## Prompting Strategy & Schemas

The generation process utilizes **zero-shot prompting** to extract semantic profiles for movies and users. The prompt templates are constructed using explicit role-based messages to carefully guide the Language Model:

- **System Message**: Sets the persona (e.g., "expert movie data analyst" or "expert user profiler") and enforces strict constraints to ensure valid JSON output (e.g., using single quotes instead of double quotes, and restricting preamble).
- **User Message**: Provides the contextual data (movie title/genres, or user's top-rated movie history) along with instructions and the expected JSON format schema.
- **Assistant Message**: Used as a generation trigger to initiate the LLM's structured response.

### Output Schemas

We enforce structured outputs by mapping the LLM's responses to specific schemas:

**1. Item (Movie) Schema**
- `plot_summary`: A concise summary of the plot (maximum 2 sentences).
- `main_themes`: Key thematic elements (e.g., friendship, betrayal).
- `mood_atmosphere`: Adjectives describing the mood or vibe.
- `target_audience`: The intended demographic.
- `visual_style`: A descriptive sentence of the movie's visual characteristics.
- `cultural_and_Language_context`: Language and cultural settings.

**2. User Schema**
- `preference_summary`: A sentence summarizing the user's overall taste.
- `favorite_genres`: The user's top 3-5 most-watched and highly rated genres, ordered by preference.
- `thematic_interests`: Preferred thematic elements inferred from their history.
- `visual_style_preferences`: Preferred visual styles based on their highly rated movies.
# LLM-enriched-MovieLens
