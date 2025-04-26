import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import lil_matrix, save_npz, load_npz


# Step 1: Parse all JSON records and track co-occurrences
item_to_related = defaultdict(float)
item_freq = defaultdict(int)
# Define relative weights
weight_bought = 1.0
weight_viewed = 0.5


with open("meta_Beauty_filter.json", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading data"):
        item = json.loads(line)
        asin = item.get("asin")
        if not asin:
            continue

        related = item.get("related", {})
        # Initialize item frequency and weighted co-occurrence dictionary
        item_freq[asin] += 1

        # Add also_bought relationships with weight
        for r in related.get("also_bought", []):
            item_to_related[(asin, r)] += weight_bought
            item_to_related[(r, asin)] += weight_bought
            item_freq[r] += 1

        # Add also_viewed relationships with weight
        for r in related.get("also_viewed", []):
            item_to_related[(asin, r)] += weight_viewed
            item_to_related[(r, asin)] += weight_viewed
            item_freq[r] += 1


# Step 2: Index mapping
unique_items = sorted(item_freq.keys())
asin_to_idx = {asin: idx for idx, asin in enumerate(unique_items)}
N = len(unique_items)

# Step 3: Sparse co-occurrence matrix
co_matrix = lil_matrix((N, N), dtype=np.float32)

for (asin1, asin2), weight in tqdm(item_to_related.items(), desc="Building matrix"):
    if asin1 in asin_to_idx and asin2 in asin_to_idx:
        i = asin_to_idx[asin1]
        j = asin_to_idx[asin2]
        co_matrix[i, j] += weight

# Step 3.5: Compute co-occurrence statistics before normalization
co_values = []

for (asin1, asin2), weight in item_to_related.items():
    if asin1 in asin_to_idx and asin2 in asin_to_idx:
        co_values.append(weight)

if co_values:
    co_values = np.array(co_values)
    print("\nRaw Co-occurrence Statistics BEFORE normalization:")
    print(f"Min:     {np.min(co_values):.4f}")
    print(f"Max:     {np.max(co_values):.4f}")
    print(f"Median:  {np.median(co_values):.4f}")
    print(f"Average: {np.mean(co_values):.4f}")
else:
    print("No co-occurrence values found.")


# Step 4: Normalize
for i in tqdm(range(N), desc="Normalizing"):
    for j in co_matrix.rows[i]:
        if co_matrix[i, j] > 0:
            co_matrix[i, j] = co_matrix[i, j] / np.sqrt(item_freq[unique_items[i]] * item_freq[unique_items[j]])

# Step 5: Save
save_npz("normalized_cooccurrence_matrix_sparse.npz", co_matrix.tocsr())
with open("asin_index.json", "w") as f:
    json.dump(unique_items, f)

# Step 6: Print sample co-occurrence scores
# Load to be sure you're using the same matrix you saved
co_matrix = load_npz("normalized_cooccurrence_matrix_sparse.npz")


# Print sample
print("\nAlso bought weights:", weight_bought)
print("Also viewed weights:", weight_viewed)

print("\nSample co-occurrence scores:")
printed = 0
for i in range(N):
    for j in co_matrix[i].nonzero()[1]:
        score = co_matrix[i, j]
        if score > 0:
            asin_i = unique_items[i]
            asin_j = unique_items[j]
            print(f"{asin_i} ↔ {asin_j}: {score:.4f}")
            printed += 1
        if printed >= 10:
            break
    if printed >= 10:
        break