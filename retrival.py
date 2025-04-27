import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import signal
import sys
import os

from get_neighbor_item import NeighborItem
class Retrieval:
    def __init__(self, alpha=0.5, lambda_=0.7, test_mode = False, k = 20, history_length = 3, retrival_method = "simple", num_workers=None, rating_normalize = "None"):
        with open('item_collaborative_similarity.json', 'r') as f:
            self.collaborative_matrix = json.load(f)           
        with open('meta_Beauty_filter.json', 'r') as f:
            self.asins = [json.loads(line)['asin'] for line in f]
        
        self.semantic_matrix = np.load('multimodal_similarity_matrix.npy')
        self.train_data = pd.read_csv('beauty.train.csv')
        self.alpha = alpha 
        self.lambda_ = lambda_
        self.test_mode = test_mode
        self.unique_items = self.train_data['asin'].unique()
        self.k = k
        self.history_length = history_length
        self.retrival_method = retrival_method
        self.num_workers = num_workers if num_workers is not None else max(mp.cpu_count() // 2, 1)
        self.rating_normalize = rating_normalize
        self.neighbor_item = NeighborItem()
        self.asin_to_index = {asin: index for index, asin in enumerate(self.asins)}

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

            # do some optimization here for searching and getting semantic score
            semantic_index_1 = self.asin_to_index[history_item]
            semantic_index_2 = self.asin_to_index[item_id]
            semantic_score = self.semantic_matrix[semantic_index_1][semantic_index_2]
            if self.rating_normalize == "None":
                score += (self.alpha * collaborative_score + (1 - self.alpha) * semantic_score) * (history_rating) * (self.lambda_ ** (i+1))
            elif self.rating_normalize == "Binary":
                score += (self.alpha * collaborative_score + (1 - self.alpha) * semantic_score) * (self.lambda_ ** (i+1))
            elif self.rating_normalize == "Centered":
                score += (self.alpha * collaborative_score + (1 - self.alpha) * semantic_score) * (history_rating - 3) * (self.lambda_ ** (i+1))
            else:
                raise ValueError(f"Invalid rating normalization: {self.rating_normalize}")
            
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
        
        if self.retrival_method == "simple":
            # simple retrival method will only use the items in collaborative matrix
            potential_items = list(set(potential_items))
            items = potential_items

        elif self.retrival_method == "neighbor":
            # neighbor retrival method will use the neighbor items of the user
            potential_items = self.neighbor_item.get_neighbor_items(user_id)
            items = potential_items
        
        elif self.retrival_method == "full":
            # full retrival method will use all items in the dataset
            items = self.unique_items
        
        elif self.retrival_method == "neighbor+simple":
            # neighbor+simple retrival method will use the neighbor items and the items in collaborative matrix
            potential_items.extend(self.neighbor_item.get_neighbor_items(user_id))
            items = list(set(potential_items))

        else:
            raise ValueError(f"Invalid retrival method: {self.retrival_method}")
        
        # Combine all scores
        scores = self._calculate_score_for_items(user_id, items)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.k]
    

if __name__ == "__main__":
    print(os.getcwd())
    try:
        retrival = Retrieval(alpha=0.5, lambda_=0.7, test_mode=True)
        print(retrival.retrieve_top_k_items("A00414041RD0BXM6WK0GX"))
    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, shutting down...')

     