from retrival import Retrieval
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_metrics(predictions, ground_truth, k):
    """
    Calculate Hits@K and NDCG@K
    Args:
        predictions: list of recommended item IDs
        ground_truth: actual item ID that user interacted with
        k: number of items to consider
    """
    # Hits@K
    hits = 1 if ground_truth in predictions[:k] else 0
    
    # NDCG@K
    dcg = 0
    idcg = 1  # Since we only have one relevant item
    for i, item in enumerate(predictions[:k]):
        if item == ground_truth:
            dcg += 1 / np.log2(i + 2)  # i + 2 because index starts from 0
    ndcg = dcg / idcg
    
    return hits, ndcg

def evaluate_model(retriever, test_data, k_values=[5, 10, 20], test_mode = False):
    """
    Evaluate retrieval model using Hits@K and NDCG@K
    """
    metrics = {k: {'hits': [], 'ndcg': []} for k in k_values}
    if test_mode:
        test_data = test_data.head(100)
    # Evaluate for each user in test set
    for _, row in tqdm(test_data.iterrows(), desc='Evaluating', total=len(test_data)):
        # Get ground truth item
        user_id = row['reviewerID']
        ground_truth = row['asin']
        
        # Get model predictions
        predictions = retriever.retrieve_top_k_items(user_id)
        pred_items = [item for item, score in predictions]
        
        # Calculate metrics for each k
        for k in k_values:
            hits, ndcg = calculate_metrics(pred_items, ground_truth, k)
            metrics[k]['hits'].append(hits)
            metrics[k]['ndcg'].append(ndcg)
    
    # Calculate average metrics
    results = {}
    for k in k_values:
        results[f'Hits@{k}'] = np.mean(metrics[k]['hits'])
        results[f'NDCG@{k}'] = np.mean(metrics[k]['ndcg'])
    
    return results

if __name__ == "__main__":
    # Initialize retriever
    retriever = Retrieval(alpha=0.5, lambda_=0.7, simple_retrival = False)
    
    # Load test data
    test_data = pd.read_csv('beauty.test.csv')
    print(f"Number of users in test set: {len(test_data['reviewerID'].unique())}")
    
    # Evaluate model
    k_values = [5, 10]
    results = evaluate_model(retriever, test_data, k_values, test_mode = True)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

