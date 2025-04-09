import pandas as pd
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import time
from tqdm import tqdm

def calculate_and_save_similarities(input_file, output_file, test_mode = False):
    """Calculate and save cosine similarities between items based on user ratings

    Calculation Example: 
    item1 is rated 5 by user1, 4 by user2, 3 by user3
    item2 is rated 4 by user2, 5 by user3, 3 by user4

    The similarity between item1 and item2 is calculated as follows:
    similarity = (5*0 + 4*4 + 3*5 + 0*3) / (sqrt(5^2 + 4^2 + 3^2) * sqrt(4^2 + 5^2 + 3^2))
    
    
    Args:
        input_file: str, path to the input file
        output_file: str, path to the output file
        test_mode: bool, whether to run in test mode, if True, only use 2000 items for testing
    
    Returns:
        similarities: dict, a dictionary of item similarities. 
                      similarities[item1][item2] = similarity between item1 and item2
    """
    start_time = time.time()
    print("Starting similarity calculation...")
    
    # Read CSV file
    df = pd.read_csv(input_file)
    n_items = len(df['asin'].unique())
    print(f"Total number of items: {n_items}")
    
    # Create item-user rating matrix
    rating_matrix = defaultdict(dict)
    for _, row in df.iterrows():
        rating_matrix[row['asin']][row['reviewerID']] = row['overall']
    
    # Get all item IDs
    items = list(rating_matrix.keys())
    
    # Calculate and save similarities
    similarities = {}

    if test_mode:
        items = items[:2000]
        print(rating_matrix['B007IY97U0'])
        print(rating_matrix['B00870XLDS'])
    # Use tqdm for better progress tracking
    for i in tqdm(range(len(items)), desc="Processing items"):
        item1 = items[i]
        similarities[item1] = {}
        
        for j in range(i + 1, len(items)):
            item2 = items[j]
            
            # Get common users
            users1 = set(rating_matrix[item1].keys())
            users2 = set(rating_matrix[item2].keys())
            common_users = users1 & users2
            
            # If no common users, similarity is 0
            if not common_users:
                continue
            
            # Get rating vectors (only for common users)
            vector1 = [rating_matrix[item1][user] for user in common_users]
            vector2 = [rating_matrix[item2][user] for user in common_users]
            vector3 = [rating_matrix[item1][user] for user in users1]
            vector4 = [rating_matrix[item2][user] for user in users2]
            # Calculate similarity
            similarity = cosine_similarity(np.array(vector1), np.array(vector2), np.array(vector3), np.array(vector4))
            
            # Only store if similarity > 0
            if similarity > 0:
                similarities[item1][item2] = float(similarity)
                if item2 not in similarities:
                    similarities[item2] = {}
                similarities[item2][item1] = float(similarity)
    
    # Save results
    save_similarities(similarities, output_file)
    
    end_time = time.time()
    print(f"\nCalculation completed in {(end_time - start_time)/60:.2f} minutes")
    print(f"Results saved to: {output_file}")
    
    return similarities

def cosine_similarity(vector1, vector2, vector3, vector4):
    """Calculate cosine similarity between two vectors"""
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        return 0
    return np.dot(vector1, vector2) / (np.linalg.norm(vector3) * np.linalg.norm(vector4))

def save_similarities(similarities, output_file):
    """Save similarity results with compression"""
    with open(output_file, 'w') as f:
        json.dump(similarities, f)
    
    # Print storage statistics
    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # Size in MB
    total_pairs = sum(len(sim_dict) for sim_dict in similarities.values())
    print(f"\nStorage statistics:")
    print(f"File size: {file_size:.2f} MB")
    print(f"Total similarity pairs stored: {total_pairs}")

def load_similarities(file_path):
    """Load similarity results"""
    print(f"Loading similarities from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def get_most_similar_items(item_id, similarities, n=5):
    """Find top N most similar items for a given item"""
    if item_id not in similarities:
        return []
    
    similar_items = sorted(similarities[item_id].items(), 
                         key=lambda x: x[1], 
                         reverse=True)
    return similar_items[:n]

def get_similarity_stats(similarities):
    """Get statistics about similarities"""
    all_sims = []
    for item_dict in similarities.values():
        all_sims.extend(item_dict.values())
    
    stats = {
        "total_pairs": len(all_sims),
        "mean_similarity": np.mean(all_sims),
        "median_similarity": np.median(all_sims),
        "min_similarity": min(all_sims),
        "max_similarity": max(all_sims)
    }
    return stats

# Usage example
if __name__ == "__main__":
    input_file = 'beauty.train.csv'
    output_file = 'item_collaborative_similarity.json'

    # Calculate and save similarities
    similarities = calculate_and_save_similarities(input_file, output_file, test_mode = False)
    
    # Print similarity statistics
    stats = get_similarity_stats(similarities)
    print("\nSimilarity Statistics:")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Mean similarity: {stats['mean_similarity']:.4f}")
    print(f"Median similarity: {stats['median_similarity']:.4f}")
    print(f"Min similarity: {stats['min_similarity']:.4f}")
    print(f"Max similarity: {stats['max_similarity']:.4f}")

    similarities = load_similarities(output_file)
    print(len(similarities.keys()))
    
    # Example: find similar items
    test_item = list(similarities.keys())[1]
    print(f"\nMost similar items for {test_item}:")
    for item, score in get_most_similar_items(test_item, similarities):
        print(f"Item ID: {item}, Similarity: {score:.4f}")