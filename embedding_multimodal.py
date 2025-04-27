import torch
import numpy as np
import json
import requests
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPModel
from transformers import ViTImageProcessor, ViTForImageClassification

from sentence_transformers import SentenceTransformer, util

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

use_multimodal = False  # Set to False to use only text embeddings
combination_method = "concat"  # "concat" or "average"

# Load models
text_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
text_model.max_seq_length = 8192

# CLIP
# image_model_name = "openai/clip-vit-base-patch32"
# image_model = CLIPModel.from_pretrained(image_model_name).to(device)
# image_processor = CLIPProcessor.from_pretrained(image_model_name)

# VIT-Transformer
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
image_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


# Load data
with open("meta_Beauty_filter.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Text prompt
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

# Image embedding
def get_image_embedding(image_url):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = image_model(pixel_values=pixel_values)

        return outputs.image_embeds.squeeze(0).to(device)
    except Exception as e:
        print(f"Failed to process image at {image_url}: {e}")
        return torch.zeros(512, device=device)


# Combine embeddings
def combine_embeddings(text_embed, image_embed, method="concat"):
    if method == "concat":
        return torch.cat((text_embed, image_embed), dim=-1).to(device)
    elif method == "average":
        return (text_embed + image_embed) / 2
    else:
        raise ValueError("Unsupported combination method")

# Generate multimodal embeddings
multimodal_embeddings = []
asin_list = []

print("Generating embeddings...")
for item in tqdm(data):
    prompt = create_prompt(item)
    text_embedding = text_model.encode(prompt, convert_to_tensor=True).to(device)

    if use_multimodal:
        image_url = item.get("imUrl", None)
        image_embedding = get_image_embedding(image_url) if image_url else torch.zeros(512, device=device)
        combined = combine_embeddings(text_embedding, image_embedding, method=combination_method)
    else:
        combined = text_embedding

    multimodal_embeddings.append(combined)

    asin_list.append(item["asin"])

multimodal_embeddings = torch.stack(multimodal_embeddings)

# Cosine similarity
print("Calculating cosine similarity...")
similarity_matrix = util.cos_sim(multimodal_embeddings, multimodal_embeddings)
out_file = "multimodal_similarity_matrix.npy" if use_multimodal else "textual_similarity_matrix.npy"
np.save(out_file, similarity_matrix.cpu().numpy().astype(np.float32))
print(f"Cosine similarity matrix saved as '{out_file}'")



# Command line control
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--multimodal", action="store_true", help="Enable multimodal (text + image) embeddings")
parser.add_argument("--method", choices=["concat", "average"], default="concat", help="Combination method")
args = parser.parse_args()

use_multimodal = args.multimodal
combination_method = args.method

## Sample usage: python embedding_multimodal.py --multimodal --method concat


