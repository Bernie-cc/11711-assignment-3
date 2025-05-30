from retrival import Retrieval
import pandas as pd
import numpy as np
from tqdm import tqdm
import signal
import sys
import multiprocessing as mp
import json
from datetime import datetime
import os

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

def evaluate_model(retriever, test_data, k_values=[5, 10, 20], test_mode=False, test_sample_size=10):
    """
    Evaluate retrieval model using Hits@K and NDCG@K
    """
    metrics = {k: {'hits': [], 'ndcg': []} for k in k_values}
    predictions_dict = {}  # Store predictions for each user
    
    if test_mode:
        test_data = test_data.head(test_sample_size)
    
    # Evaluate for each user in test set
    for _, row in tqdm(test_data.iterrows(), desc='Evaluating', total=len(test_data)):
        user_id = row['reviewerID']
        ground_truth = row['asin']
        
        # Get model predictions
        predictions = retriever.retrieve_top_k_items(user_id)
        pred_items = [item for item, score in predictions]
        
        # Store predictions
        predictions_dict[user_id] = {
            'retrieved_items': predictions,  # List of (item_id, score) tuples
            'ground_truth': ground_truth
        }
        
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
    
    return results, predictions_dict

def signal_handler(sig, frame):
    print('\nGracefully shutting down...')
    # Terminate any active multiprocessing pools
    if hasattr(mp, '_current_process') and mp._current_process()._pool is not None:
        mp._current_process()._pool.terminate()
    sys.exit(0)

def log_experiment(results, retriever, test_data, predictions_dict, log_dir='experiment_logs'):
    """
    Log experiment parameters, results and detailed retrieval results
    
    Will generate two files:
    1. experiment_metrics.jsonl: Append each experiment's metrics
       Format:
       {
           "timestamp": "2024-03-15 10:30:45",
           "parameters": {
               "alpha": 0.5,
               "lambda": 0.7,
               "history_length": 3,
               "k": 20,
               "simple_retrival": true,
               "num_workers": 4
           },
           "results": {
               "Hits@5": 0.123,
               "NDCG@5": 0.456,
               "Hits@10": 0.234,
               "NDCG@10": 0.567
           }
       }
    
    2. retrieval_results_{timestamp}.json: Detailed retrieval results for each user
       Format:
       {
           "parameters": {
               // same as above parameters
           },
           "predictions": {
               "user_id_1": {
                   "retrieved_items": [
                       ["item_id_1", 0.95],  // [item_id, score]
                       ["item_id_2", 0.85],
                       ...
                   ],
                   "ground_truth": "true_item_id"
               },
               "user_id_2": {
                   // same structure as above
               }
           }
       }
    """
    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Log experiment parameters and metrics
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'alpha': retriever.alpha,
            'lambda': retriever.lambda_,
            'history_length': retriever.history_length,
            'k': retriever.k,
            'simple_retrival': retriever.simple_retrival,
            'num_workers': retriever.num_workers,
            'rating_normalize': retriever.rating_normalize
        },
        'results': results
    }
    
    # Save metrics log
    metrics_file = os.path.join(log_dir, 'experiment_metrics.jsonl')
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Save detailed retrieval results
    retrieval_file = os.path.join(log_dir, f'retrieval_results_{timestamp}.json')
    retrieval_results = {
        'parameters': log_entry['parameters'],
        'predictions': predictions_dict
    }
    with open(retrieval_file, 'w') as f:
        json.dump(retrieval_results, f, indent=2)
    
    print(f"\nExperiment metrics logged to {metrics_file}")
    print(f"Retrieval results saved to {retrieval_file}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        test_sample_size = 100
        # Initialize retriever
        retriever = Retrieval(alpha=0.5, lambda_=0.7, simple_retrival=True, rating_normalize="Centered")
        
        # Load test data
        test_data = pd.read_csv('beauty.test.csv')
        print(f"Number of users in test set: {len(test_data['reviewerID'].unique())}")
        
        # Evaluate model
        k_values = [5, 10]
        results, predictions_dict = evaluate_model(retriever, test_data, k_values, test_mode=True, test_sample_size=test_sample_size)
        
        # Print results
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        # Log experiment
        log_experiment(results, retriever, test_data, predictions_dict)
            
    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, shutting down...')

