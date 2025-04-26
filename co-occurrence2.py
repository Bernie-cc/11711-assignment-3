import json
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import lil_matrix, save_npz, load_npz
from pathlib import Path

# Define relative weights for metadata-based co-occurrence
weight_bought = 1.0
weight_viewed = 1.0

def build_cooccurrence_from_json(json_file):
    item_to_related = defaultdict(float)
    item_freq = defaultdict(int)

    with open(json_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading JSON data"):
            item = json.loads(line)
            asin = item.get("asin")
            if not asin:
                continue

            related = item.get("related", {})
            item_freq[asin] += 1

            for r in related.get("also_bought", []):
                item_to_related[(asin, r)] += weight_bought
                item_to_related[(r, asin)] += weight_bought
                item_freq[r] += 1

            for r in related.get("also_viewed", []):
                item_to_related[(asin, r)] += weight_viewed
                item_to_related[(r, asin)] += weight_viewed
                item_freq[r] += 1
    
    return item_to_related, item_freq

def build_cooccurrence_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by=["reviewerID", "unixReviewTime"])

    item_to_related = defaultdict(float)
    item_freq = defaultdict(int)

    for _, group in tqdm(df.groupby("reviewerID"), desc="Processing user sequences"):
        sequence = list(group["asin"])
        for i in range(1, len(sequence)):
            current = sequence[i]
            prev_items = sequence[:i]

            item_freq[current] += 1
            for p in prev_items:
                item_to_related[(current, p)] += 1
                item_to_related[(p, current)] += 1
                item_freq[p] += 1
    
    return item_to_related, item_freq

def compute_and_save_matrix(item_to_related, item_freq, output_prefix):
    output_file = output_prefix + "collaborative_similarity.json"
    unique_items = sorted(item_freq.keys())
    asin_to_idx = {asin: idx for idx, asin in enumerate(unique_items)}
    N = len(unique_items)

    co_matrix = lil_matrix((N, N), dtype=np.float32)

    for (asin1, asin2), weight in tqdm(item_to_related.items(), desc="Building matrix"):
        if asin1 in asin_to_idx and asin2 in asin_to_idx:
            i = asin_to_idx[asin1]
            j = asin_to_idx[asin2]
            co_matrix[i, j] = weight

    # Remove outliers (optional)
    raw_weights = np.array(list(item_to_related.values()))
    if raw_weights.size > 0:
        threshold = np.percentile(raw_weights, 99)
        max_inlier = raw_weights[raw_weights <= threshold].max()
        for key in item_to_related:
            if item_to_related[key] > threshold:
                item_to_related[key] = max_inlier
        print(f"Outliers capped to {max_inlier:.4f}")

    # Normalize and build similarities dictionary
    print("Normalizing and building similarity dictionary...")
    similarities = defaultdict(dict)

    for i in range(N):
        for j in co_matrix.rows[i]:
            if co_matrix[i, j] > 0:
                denom = np.sqrt(item_freq[unique_items[i]] * item_freq[unique_items[j]])
                if denom == 0:
                    continue
                norm_score = co_matrix[i, j] / denom
                asin_i = unique_items[i]
                asin_j = unique_items[j]
                similarities[asin_i][asin_j] = round(float(norm_score), 6)

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(similarities, f)
    with open("asin_index.json", "w") as f:
        json.dump(unique_items, f)

    print(f"\nSample similarities:")
    printed = 0
    for asin, sim_dict in similarities.items():
        for other, score in sim_dict.items():
            print(f"{asin} ↔ {other}: {score:.4f}")
            printed += 1
            if printed >= 10:
                break
        if printed >= 10:
            break
    print(f"\nSimilarity JSON saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to meta_Beauty_filter.json or beauty.train.csv")
    args = parser.parse_args()

    ext = Path(args.input).suffix.lower()
    output_prefix = ""

    if ext == ".json":
        item_to_related, item_freq = build_cooccurrence_from_json(args.input)
        output_prefix = "meta_"
    elif ext == ".csv":
        item_to_related, item_freq = build_cooccurrence_from_csv(args.input)
        output_prefix = "rating_"
    else:
        raise ValueError("Unsupported file type. Use a .json or .csv file.")

    compute_and_save_matrix(item_to_related, item_freq, output_prefix)


# Usage example:
    # For metadata-based similarity
    # python co-occurrence2.py --input meta_Beauty_filter.json

    # For sequence-based co-occurrence
    # python co-occurrence2.py --input beauty.train.csv

