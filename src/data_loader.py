import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

DATASET_URLS = {
    '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
}

def download_dataset(dataset: str, data_dir: str):
    """Downloads and extracts the dataset if it doesn't already exist."""
    dataset_dir = os.path.join(data_dir, f'ml-{dataset}')
    if os.path.exists(dataset_dir):
        print(f"Dataset ml-{dataset} already exists at {dataset_dir}. Skipping download.")
        return
    
    url = DATASET_URLS.get(dataset)
    if not url:
        raise ValueError(f"Unknown dataset '{dataset}'")
        
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f'ml-{dataset}.zip')
    
    print(f"Downloading ml-{dataset} from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    print(f"Dataset ml-{dataset} ready.")

def load_data(dataset: str, data_dir: str):
    """Loads interaction data and item metadata for the given dataset."""
    if dataset == '100k':
        interactions_path = os.path.join(data_dir, 'ml-100k', 'u.data')
        items_path = os.path.join(data_dir, 'ml-100k', 'u.item')
        
        # Load interactions
        df = pd.read_csv(interactions_path, sep='\t', names=['userID', 'itemID', 'rating', 'timestamp'])
        
        # Load items
        item_headers = ["itemID", "title", "release_date", "video_release_date", "IMDb_URL", 
                        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
                        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
                        "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        items_df = pd.read_csv(items_path, sep='|', names=item_headers, encoding='latin-1')
        
        # Merge basic item metadata into interactions for ease of use
        df = df.merge(items_df[['itemID', 'title']], on='itemID', how='left')
        
    elif dataset == '1m':
        interactions_path = os.path.join(data_dir, 'ml-1m', 'ratings.dat')
        items_path = os.path.join(data_dir, 'ml-1m', 'movies.dat')
        
        # Load interactions
        df = pd.read_csv(interactions_path, sep='::', engine='python', names=['userID', 'itemID', 'rating', 'timestamp'])
        
        # Load items
        item_headers = ["itemID", "title", "genres"]
        items_df = pd.read_csv(items_path, sep='::', engine='python', names=item_headers, encoding='latin-1')
        
        # Merge basic item metadata into interactions
        df = df.merge(items_df[['itemID', 'title']], on='itemID', how='left')
        
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
        
    return df, items_df

def apply_cold_start_split(dataset: str, df: pd.DataFrame, items_df: pd.DataFrame, seed: int = 42):
    """Applies the 80/20 warm/cold split and filters users."""
    print("\n=== Applying Cold-Start Strategy Split ===")
    all_items = df['itemID'].unique()
    
    np.random.seed(seed)
    cold_items = np.random.choice(all_items, size=int(len(all_items) * 0.2), replace=False)
    warm_items = np.setdiff1d(all_items, cold_items)
    
    print(f"Total Items: {len(all_items)} | Warm: {len(warm_items)} | Cold: {len(cold_items)}")
    
    # Create Warm Interactions DataFrame (Remove all cold item interactions)
    warm_df = df[df['itemID'].isin(warm_items)].copy()
    
    # Filter Users: Must have >= 5 warm interactions to build a profile
    user_counts = warm_df.groupby('userID').size()
    valid_users = user_counts[user_counts >= 5].index
    warm_df = warm_df[warm_df['userID'].isin(valid_users)]
    
    print(f"Total Users original: {df['userID'].nunique()} | Valid Users: {len(valid_users)}")
    
    # We only use highly rated WARM movies to build the user profile
    high_rated_warm_df = warm_df[warm_df['rating'] >= 4].copy()
    
    return warm_df, valid_users, high_rated_warm_df, cold_items

def build_movie_inputs(dataset: str, items_df: pd.DataFrame) -> list:
    """Builds the list of dictionaries expected by the movie profile generation chain."""
    movie_batch_inputs = []
    
    if dataset == '100k':
        genre_cols = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
                      "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
                      "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        unique_items = items_df[['itemID', 'title'] + genre_cols].drop_duplicates() 
        
        for _, row in unique_items.iterrows():
            g_list = [g for g in genre_cols if row.get(g, 0) == 1]
            movie_batch_inputs.append({
                "item_id": int(row['itemID']),
                "title": row['title'],
                "genres": ", ".join(g_list)
            })
            
    elif dataset == '1m':
        unique_items = items_df[['itemID', 'title', 'genres']].drop_duplicates()
        for _, row in unique_items.iterrows():
            genres_str = row['genres'].replace('|', ', ')
            movie_batch_inputs.append({
                "item_id": int(row['itemID']),
                "title": row['title'],
                "genres": genres_str
            })
            
    return movie_batch_inputs
