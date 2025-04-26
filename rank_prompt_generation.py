import json
import pandas as pd

# Step 1: Load retrieval_results.json
with open("user_results.json", "r") as f:
    retrieval_results = json.load(f)

# Step 2: Load beauty.train.csv
beauty_train = pd.read_csv("beauty.train.csv")

# Step 3: Load meta_Beauty_filter.json
meta_beauty = {}
with open("meta_Beauty_filter.json", "r") as f:
    for line in f:
        product = json.loads(line)
        meta_beauty[product["asin"]] = product

# Initialize a dictionary to store prompts
reviewer_prompts = {}

# Step 4: Process each reviewerID
for reviewer_id, items in retrieval_results.items():
    # Get the most recent 3 purchases for the reviewer
    user_purchases = beauty_train[beauty_train["reviewerID"] == reviewer_id]
    recent_purchases = user_purchases.sort_values(by="unixReviewTime", ascending=False).head(3)

    # Formulate the user prompt
    user_prompt = f"User: {reviewer_id} has purchased the following items in this order:\n"
    for _, row in recent_purchases.iterrows():
        asin = row["asin"]
        if asin in meta_beauty:
            product = meta_beauty[asin]
            user_prompt += (
                f'{{ "ItemID": {asin}, "title": "{product.get("title", "N/A")}", '
                f'"salesRank_Beauty": {product.get("salesRank", {}).get("Beauty", "N/A")}, '
                f'"categories": {product.get("categories", "N/A")}, '
                f'"price": {product.get("price", "N/A")}, '
                f'"brand": "{product.get("brand", "N/A")}" }},\n'
            )

    user_prompt = user_prompt.rstrip(",\n") + "\n\nI will provide you with 20 items, each indicated by number identifier[]. Analyze the user’s purchase history to identify preferences and purchase patterns. Then, rank the candidate items based on their alignment with the user’s preferences and other contextual factors.\n"

    user_prompt2 = ""
    # Add the 4 items to the user prompt
    for idx, asin in enumerate(items, start=1):
        if asin in meta_beauty:
            product = meta_beauty[asin]     
            user_prompt2 += (
                f"[{idx}] {{ \"title\": \"{product.get('title', 'N/A')}\", "
                f"\"salesRank_Beauty\": {product.get('salesRank', {}).get('Beauty', 'N/A')}, "
                f"\"categories\": {product.get('categories', 'N/A')}, "
                f"\"price\": {product.get('price', 'N/A')}, "
                f"\"brand\": \"{product.get('brand', 'N/A')}\",}}\n"  
                )

    user_prompt3 =  "\nAnalyze the user’s purchase history to identify user preferences and purchase patterns. Then, rank the 20 items above based on their alignment with the user’s preferences and other contextual factors. All the items should be included and listed using identifiers, in descending order of the user’s preference. The most preferred recommendation item should be listed first. The output format should be []>[], where each [] is an identifier, e.g., [1]>[2]. Only respond with the ranking results, do not say any word or explain. Output in the following JSON format:\n\n{ \"rank\": \"[]>[]..>[]\" }\n"

    # Structure the prompt with roles
    prompt = [
        {"role": "system", "content": "You are an intelligent assistant that can rank items based on the user’s preference."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Okay,please provide the items."},
        {"role": "user", "content": user_prompt2},
        {"role": "assistant", "content": "Received 20 item."},
        {"role": "user", "content": user_prompt3},

    ]

    # Store the prompt in the dictionary
    reviewer_prompts[reviewer_id] = prompt

# Save all prompts to a JSON file
with open("reviewer_prompts.json", "w", encoding="utf-8") as f:
    json.dump(reviewer_prompts, f, indent=4)