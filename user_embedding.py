import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pandas as pd

# Initialize the embedding model
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
model.max_seq_length = 8192

print("Loading data...")
# Load the dataset
data = pd.read_csv('beauty.train.csv')
with open("meta_Beauty_filter.json", "r", encoding="utf-8") as f:
    items_all = [json.loads(line) for line in f]
# Extract reviewerID and asin (user review history)
user_history = (
    data.groupby('reviewerID')
    .apply(lambda x: x.sort_values(by='unixReviewTime', ascending=False)['asin'].tolist())
    .to_dict()
)

# Combine relevant fields into a single text prompt for each item
def item_prompt(item):
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
        f"price: {price}\n"
        )


# Generate embeddings for all items
print("Generating prompts...")
prompts = []
users= []
print("Number of users:", len(user_history))
for user, items in tqdm(user_history.items(), desc="Processing users"):
    prompt = [f"User {user} has purchased the following items in the most recent order:\n"]
    for i, item in enumerate(items):
        # Find the corresponding item in the dataset
        item_data = next((item_data for item_data in items_all if item_data['asin'] == item), None)
        # If the item is found, create a prompt for it
        if item_data:
            prompt.append(f"[{i}]"+item_prompt(item_data))
    prompts.append("\n".join(prompt))
    users.append(user)

print(prompts[0])  # Print the first prompt for debugging
print("Number of users:", len(prompts))
print("Generating embeddings...")
embeddings = model.encode(prompts,batch_size=6, show_progress_bar=True, device="cuda")

# Extract ASINs and calculate pairwise cosine similarity
def calculate_and_save_user_similarities(embeddings, output_file):
    print("Calculating user similarities...")
    # Calculate the cosine similarity matrix
    similarities_matrix= model.similarity(embeddings, embeddings)
    print("Cosine similarity matrix shape:", similarities_matrix.shape)
    np.save(output_file, similarities_matrix.numpy())

# Calculate and save ASIN similarities
output_file = "users_similarities_matrix.npy"
calculate_and_save_user_similarities(embeddings, output_file)
# Save the user_history to a json file
with open("user_history.json", "w", encoding="utf-8") as f:
    json.dump(user_history, f, ensure_ascii=False, indent=4)