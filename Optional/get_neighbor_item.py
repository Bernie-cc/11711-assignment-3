import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import signal
import sys
from collections import OrderedDict
from typing import List, Set, Dict


class NeighborItem:
    def __init__(self, top_k = 10):
        with open('user_history.json', 'r') as f:
            self.user_history = json.load(f,object_pairs_hook=OrderedDict)
        
        self.users = list(self.user_history.keys())
        self.user_to_index = {user: idx for idx, user in enumerate(self.users)}

        self.users_similarities_matrix = np.load('users_similarities_matrix.npy')
        self.top_k = top_k

        # cache for neighbors and items
        self._neighbor_cache: Dict[str, List[str]] = {}
        self._items_cache: Dict[str, List[str]] = {}

    def get_neighbors(self, user_id) -> List[str]:
        '''
        get the top k neighbors of the user
        return: list of neighbors real user id
        '''
        if user_id not in self._neighbor_cache:
            user_index = self.user_to_index[user_id]
            similarities = self.users_similarities_matrix[user_index]
            indices = np.argpartition(similarities, -self.top_k)[-self.top_k:]
            indices = indices[np.argsort(similarities[indices])][::-1]
            neighbors = [self.users[i] for i in indices]
            self._neighbor_cache[user_id] = neighbors
        return self._neighbor_cache[user_id]
    
    def get_neighbor_items(self, user_id) -> List[str]:
        '''
        get the neighbor items of the user
        return: list of neighbor items asin 
        ''' 
        if user_id not in self._items_cache:
            neighbors = self.get_neighbors(user_id)
            neighbor_items = []
            for neighbor in neighbors:
                neighbor_items.extend(self.user_history[neighbor])
            neighbor_items = list(set(neighbor_items))
            self._items_cache[user_id] = neighbor_items
        return self._items_cache[user_id]
    
    