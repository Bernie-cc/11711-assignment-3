import json
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm

# Initialize the embedding model
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
model.max_seq_length = 8192

# Load data from meta_Beauty_filter.json
print("Loading data...")
with open("meta_Beauty_filter.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Combine relevant fields into a single text prompt for each item
def create_prompt(item):
    title = item.get('title', 'N/A')
    description = item.get('description', 'N/A')
    categories = ' > '.join(item['categories'][0]) if 'categories' in item and item['categories'] else 'N/A'
    brand = item.get('brand', 'N/A')
    sales_rank = item.get('salesRank', {}).get('Beauty', 'N/A')
    price = item.get('price', 'N/A')

    return (
        f"title: {title}\n"
        f"description: {description}\n"
        f"categories: {categories}\n"
        f"brand: {brand}\n"
        f"salesRank: {sales_rank}\n"
        f"price: {price}"
        )

# Generate embeddings for all items
print("Generating prompts...")
prompts = [create_prompt(item) for item in tqdm(data)]
print(prompts[0])  # Print the first prompt for debugging
print("Number of items:", len(prompts))
print("Number of unique ASINs:", len(set(item['asin'] for item in data)))
print("Generating embeddings...")
embeddings = model.encode(prompts,batch_size=8, show_progress_bar=True, device="cuda")

# Extract ASINs and calculate pairwise cosine similarity
def calculate_and_save_asin_similarities(embeddings, output_file):
    print("Calculating ASIN similarities...")
    # Calculate the cosine similarity matrix
    similarities_matrix= model.similarity(embeddings, embeddings)
    print("Cosine similarity matrix shape:", similarities_matrix.shape)
    np.save(output_file, similarities_matrix.numpy())

# Calculate and save ASIN similarities
output_file = "similarities_matrix.npy"
calculate_and_save_asin_similarities(embeddings, output_file)