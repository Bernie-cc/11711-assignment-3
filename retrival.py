import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import signal
import sys

class Retrieval:
    def __init__(self, alpha=0.5, lambda_=0.7, test_mode = False, k = 20, history_length = 3, simple_retrival = False, num_workers=None):
        with open('item_collaborative_similarity.json', 'r') as f:
            self.collaborative_matrix = json.load(f)           
        with open('meta_Beauty_filter.json', 'r') as f:
            self.asins = [json.loads(line)['asin'] for line in f]
        
        self.semantic_matrix = np.load('similarities_matrix.npy')
        self.train_data = pd.read_csv('beauty.train.csv')
        self.alpha = alpha 
        self.lambda_ = lambda_
        self.test_mode = test_mode
        self.unique_items = self.train_data['asin'].unique()
        self.k = k
        self.history_length = history_length
        self.simple_retrival = simple_retrival
        self.num_workers = num_workers if num_workers is not None else max(mp.cpu_count() // 2, 1)
    def get_history_interaction(self, user_id):
        '''
        Get the history interaction of a user include tuple of (item, rating)
        Args:
            user_id: user ID to get history for
        Returns:
            list: list of tuples (item_id, rating) that the user has interacted with in descending order of timestamp
        '''
        # Get user's interactions and sort by timestamp
        user_history = self.train_data[self.train_data['reviewerID'] == user_id]
        user_history = user_history.sort_values('unixReviewTime', ascending=False)
        user_history = user_history.head(self.history_length)
        # Convert to list of tuples (item_id, rating)
        return list(zip(user_history['asin'], user_history['overall']))
    
    def calculate_retrival_score(self, user_id, item_id):
        history_interaction = self.get_history_interaction(user_id)
        n = len(history_interaction)

        score = 0
        for i, (history_item, history_rating) in enumerate(history_interaction):
            if history_item == item_id:
                continue

            collaborative_score = self.get_collaborative_score(item_id, history_item)
            semantic_index_1 = self.asins.index(history_item)
            semantic_index_2 = self.asins.index(item_id)
            semantic_score = self.semantic_matrix[semantic_index_1][semantic_index_2]
            score += (self.alpha * collaborative_score + (1 - self.alpha) * semantic_score) * (history_rating) * (self.lambda_ ** (i+1))
        
        score = score / n if n > 0 else 0
        return score
    
    def get_collaborative_score(self, item_id, history_item):
        """
        Get collaborative filtering similarity score between two items0
        Args:
            item_id: target item ID
            history_item: historical item ID
        Returns:
            float: similarity score, returns 0 if either item not in matrix
        """
        try:
            # Check if item_id exists in matrix
            if item_id not in self.collaborative_matrix:
                return 0
            # Check if history_item exists in item_id's similarities
            if history_item not in self.collaborative_matrix[item_id]:
                return 0
            # Return the similarity score
            return self.collaborative_matrix[item_id][history_item]
        except:
            # Return 0 for any other errors
            return 0
    
    def _calculate_score_for_items(self, user_id, items):
        scores = []
        for item in tqdm(items, desc=f"Worker processing {len(items)} items", leave=False):
            score = self.calculate_retrival_score(user_id, item)
            scores.append((item, score))
        return scores

    def retrieve_top_k_items(self, user_id):
        scores = []
        history_interaction = self.get_history_interaction(user_id)
        potential_items = []
        for item, rating in history_interaction:
            potential_items.extend(self.collaborative_matrix[item].keys())
        
        if self.simple_retrival:
            potential_items = list(set(potential_items))
            items = potential_items
        else:
            items = self.unique_items

        # Split items into chunks for parallel processing
        num_cores = self.num_workers
        print(f"Processing {len(items)} items using {num_cores} cores...")
        chunk_size = len(items) // num_cores
        item_chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Create a pool of workers
        pool = mp.Pool(num_cores)
        try:
            partial_func = partial(self._calculate_score_for_items, user_id)
            chunk_scores = list(tqdm(
                pool.imap(partial_func, item_chunks),
                total=len(item_chunks),
                desc=f"Processing user {user_id}"
            ))
        finally:
            pool.close()
            pool.join()
        
        # Combine all scores
        scores = [item for sublist in chunk_scores for item in sublist]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.k]
    
def signal_handler(sig, frame):
    print('\nGracefully shutting down...')
    # 确保关闭所有进程池
    if hasattr(mp, '_current_process') and mp._current_process()._pool is not None:
        mp._current_process()._pool.terminate()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        retrival = Retrieval(alpha=0.5, lambda_=0.7, test_mode=True)
        print(retrival.retrieve_top_k_items("A00414041RD0BXM6WK0GX"))
    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, shutting down...')

     