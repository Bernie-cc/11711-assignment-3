import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json
import numpy as np

with open("meta_Beauty_filter.json", "r", encoding="utf-8") as f:
    asins = [json.loads(line)['asin'] for line in f]
    similarities_matrix=np.load("similarities_matrix.npy")
    similarities = defaultdict(dict)
    for i in tqdm(range(len(asins))):
        for j in range(i + 1, len(asins)):
            similarity = float(similarities_matrix[i, j])  # Convert to Python float
            similarities[asins[i]][asins[j]] = similarity
            similarities[asins[j]][asins[i]] = similarity
    
# Save results to a JSON file
with open('item_semantic_similarity.json', "w", encoding="utf-8") as f:
    json.dump(similarities, f)



