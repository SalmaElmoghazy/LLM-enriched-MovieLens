import os
import argparse
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from src.data_loader import download_dataset, load_data, apply_cold_start_split, build_movie_inputs
from src.chains import build_chains

# Constants
BATCH_SIZE = 1
SAVE_EVERY_N = 16
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate MovieLens Profiles using LLM")
    parser.add_argument('--dataset', type=str, choices=['100k', '1m'], required=True, help="Dataset to process (100k or 1m)")
    parser.add_argument('--data-dir', type=str, default='data', help="Directory to download/store the dataset (default: 'data')")
    return parser.parse_args()

def setup_dirs(dataset: str, data_dir_arg: str):
    data_dir = os.path.join(os.getcwd(), data_dir_arg) if not os.path.isabs(data_dir_arg) else data_dir_arg
    results_dir = os.path.join(os.getcwd(), 'results', dataset)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir

def save_checkpoint(data: dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"--> Checkpoint saved: {len(data)} items to {filepath}")

def load_checkpoint(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {}

def load_model():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set or empty. Please set it to proceed.")
        
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading Model: {model_id} (Unquantized BF16)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        return_full_text=False
    )
    
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    movie_chain, user_chain = build_chains(llm)
    return movie_chain, user_chain

def generate_movie_profiles(movie_chain, movie_inputs, results_dir):
    filepath = os.path.join(results_dir, "movie_profiles.json")
    processed_movies = load_checkpoint(filepath)
    
    # Filter out already processed movies
    remaining_inputs = [m for m in movie_inputs if str(m['item_id']) not in processed_movies]
    
    if not remaining_inputs:
        print("All movie profiles already generated.")
        return
        
    print(f"\n=== Starting Movie Profile Generation ({len(remaining_inputs)} remaining) ===")
    
    for i in tqdm(range(0, len(remaining_inputs), BATCH_SIZE), desc="Movies"):
        batch = remaining_inputs[i : i + BATCH_SIZE]
        try:
            results = movie_chain.batch(batch)
            
            for original_input, res in zip(batch, results):
                if isinstance(res, dict):
                    year_match = re.search(r'\((\d{4})\)', original_input['title'])
                    extracted_year = year_match.group(1) if year_match else "Unknown"
                    
                    full_profile = {
                        "item_id": original_input["item_id"],
                        "title": original_input["title"],
                        "year": extracted_year,
                        "genres": original_input["genres"].split(", ") if original_input["genres"] else [],
                        **res
                    }
                    
                    key = str(full_profile['item_id'])
                    processed_movies[key] = full_profile
                    
        except Exception as e:
            print(f"Batch Error: {e}")

        if (i + BATCH_SIZE) % SAVE_EVERY_N == 0:
            save_checkpoint(processed_movies, filepath)

    save_checkpoint(processed_movies, filepath)

def generate_user_profiles(user_chain, warm_df, high_rated_warm_df, valid_users, results_dir):
    filepath = os.path.join(results_dir, "user_profiles.json")
    processed_users = load_checkpoint(filepath)
    
    user_batch_inputs = []
    for uid in valid_users:
        if str(uid) in processed_users:
            continue
            
        user_history = high_rated_warm_df[high_rated_warm_df['userID'] == uid].sort_values('rating', ascending=False).head(10)
        
        if user_history.empty:
             user_history = warm_df[warm_df['userID'] == uid].sort_values('rating', ascending=False).head(5)
             
        history_str = "\n".join([f"- {row['title']} ({row['rating']})" for _, row in user_history.iterrows()])
        user_batch_inputs.append({"user_id": int(uid), "user_history": history_str})
        
    if not user_batch_inputs:
        print("All user profiles already generated.")
        return
        
    print(f"\n=== Starting User Profile Generation ({len(user_batch_inputs)} remaining) ===")
        
    for i in tqdm(range(0, len(user_batch_inputs), BATCH_SIZE), desc="Users"):
        batch = user_batch_inputs[i : i + BATCH_SIZE]
        try:
            results = user_chain.batch(batch)
            for res in results:
                if isinstance(res, dict) and 'user_id' in res:
                    processed_users[str(res['user_id'])] = res
        except Exception as e:
             print(f"Batch Error: {e}")

        if (i + BATCH_SIZE) % SAVE_EVERY_N == 0:
             save_checkpoint(processed_users, filepath)
             
    save_checkpoint(processed_users, filepath)

def main():
    args = parse_args()
    dataset = args.dataset
    
    data_dir, results_dir = setup_dirs(dataset, args.data_dir)
    
    # Download and load data
    download_dataset(dataset, data_dir)
    df, items_df = load_data(dataset, data_dir)
    
    # Apply cold start split
    warm_df, valid_users, high_rated_warm_df, cold_items = apply_cold_start_split(dataset, df, items_df)
    
    # Save cold items
    cold_items_path = os.path.join(results_dir, "cold_item_ids.npy")
    np.save(cold_items_path, cold_items)
    print(f"Saved cold_items to {cold_items_path}")
    
    # Load model and chains
    movie_chain, user_chain = load_model()
    
    # Generate movie profiles
    movie_inputs = build_movie_inputs(dataset, items_df)
    generate_movie_profiles(movie_chain, movie_inputs, results_dir)
    
    # Generate user profiles
    generate_user_profiles(user_chain, warm_df, high_rated_warm_df, valid_users, results_dir)

if __name__ == "__main__":
    main()
