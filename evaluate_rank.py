import json
import numpy as np

def evaluate_rank_results(rank_results, retrieval_results, k_values=[5, 10, 20]):
    """
    Evaluate rank results using Hits@K and NDCG@K.
    Args:
        rank_results: Dictionary containing rank results for each user.
        retrieval_results: Dictionary containing retrieval results for each user.
        k_values: List of K values to evaluate.
    Returns:
        Dictionary containing Hits@K and NDCG@K for rank results.
    """
    metrics = {k: {'hits': [], 'ndcg': []} for k in k_values}

    for user_id, rank_data in rank_results.items():
        if user_id not in retrieval_results:
            continue
        
        # Ensure rank_data is a dictionary
        if isinstance(rank_data, str):
            rank_data = json.loads(rank_data)
        
        # Sanitize and parse rank order
        rank_string = rank_data['rank'].replace('[', '').replace(']', '').replace(' ', '')  # Remove brackets and spaces
        rank_order = list(map(int, filter(None, rank_string.split('>'))))  # Split, filter empty strings, and convert to integers
        retrieved_items = retrieval_results[user_id]['retrieved_items']
        reordered_items = [retrieved_items[i - 1][0] for i in rank_order]  # Adjust index (1-based to 0-based)
        ground_truth = retrieval_results[user_id]['ground_truth']
        
        # Calculate metrics for each k
        for k in k_values:
            hits, ndcg = calculate_metrics(reordered_items, ground_truth, k)
            metrics[k]['hits'].append(hits)
            metrics[k]['ndcg'].append(ndcg)
    
    # Calculate average metrics
    results = {}
    for k in k_values:
        results[f'Hits@{k}'] = np.mean(metrics[k]['hits'])
        results[f'NDCG@{k}'] = np.mean(metrics[k]['ndcg'])
    
    return results

def calculate_metrics(predictions, ground_truth, k):
    """
    Calculate Hits@K and NDCG@K.
    Args:
        predictions: List of recommended item IDs.
        ground_truth: Actual item ID that user interacted with.
        k: Number of items to consider.
    Returns:
        Tuple containing Hits@K and NDCG@K.
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

def main(rank_results_file, retrieval_results_file, k_values=[5, 10, 20]):
    """
    Main function to evaluate rank results from input files.
    Args:
        rank_results_file: Path to the JSON file containing rank results.
        retrieval_results_file: Path to the JSON file containing retrieval results.
        k_values: List of K values to evaluate.
    """
    # Load rank results
    with open(rank_results_file, 'r') as f:
        rank_results = json.load(f)
    
    # Load retrieval results
    with open(retrieval_results_file, 'r') as f:
        retrieval_results = json.load(f)['predictions']
    
    # Evaluate rank results
    results = evaluate_rank_results(rank_results, retrieval_results, k_values)
    
    # Print results
    print("\nEvaluation Results for Rank Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main("rank_results.json", "retrieval_results.json", k_values=[5, 10, 20])
