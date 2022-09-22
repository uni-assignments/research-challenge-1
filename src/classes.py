import numpy as np
import time

class Id_Converter:
    """Contains all relevant info about the element"""
    def __init__(self):    
        """Converts string ids to unique integer ids and stores relation in a dictionary"""
        self.user_ids = {}
        self.item_ids = {}

    def get_number_of_users(self) -> int:
        """Returns the number of users"""
        return len(self.user_ids)
    
    def get_number_of_items(self) -> int:
        """Returns the number of items"""
        return len(self.item_ids)

    def add_id(self, type: str, old_id: str) -> int:
        """Checks if the id is already in the dictionary, if not adds it and returns the new id"""
        
        if type == 'User':
            if old_id not in self.user_ids:
                self.user_ids[old_id] = len(self.user_ids)
            return self.user_ids[old_id]
        else:
            if old_id not in self.item_ids:
                self.item_ids[old_id] = len(self.item_ids)
            return self.item_ids[old_id]
        

    def convert_user(self, old_id: str) -> int:
        """Converts the string id to the integer id"""
        return self.user_ids[old_id]
    
    def convert_item(self, old_id: str) -> int:
        """Converts the string id to the integer id"""
        return self.item_ids[old_id]

class Items:
    """Contains all relevant info about items"""
    def __init__(self):    
        self.item_has_user = {}
        self.user_has_item = {}
        self._pairs = {}

        self.item_ratings = {}
        self.item_mean_ratings = {}

        self.most_similar = {}


    def add(self, item_id: int, user_id: int, rating: float) -> None:
        """Adds a rating to the list of ratings"""
        if item_id not in self.item_user_pairs:
            self.item_user_pairs[item_id] = {}
            self.item_ratings[item_id] = []

        if user_id not in self.user_item_pairs:
            self.user_item_pairs[user_id] = {}

        self.item_user_pairs[item_id][user_id] = rating
        self.user_item_pairs[user_id][item_id] = rating

        self.item_ratings[item_id].append(rating)

    def get_rating(self, item_id: int, user_id: int) -> int:
        """Returns the rating of the item"""
        return self.item_user_pairs[item_id][user_id]

    def get_pairs_for_item(self, item_id: int) -> dict:
        """Returns the user-item pairs"""
        return self.item_user_pairs[item_id]
    
    def get_pairs_for_user(self, user_id: int) -> dict:
        """Returns the user-item pairs"""
        return self.user_item_pairs[user_id]
    
    def compute_mean(self) -> None:
        """Computes the mean of all ratings"""
        for item_id, ratings in self.item_ratings.items():
            self.item_mean_ratings[item_id] = sum(ratings)/len(ratings)
    
    def get_mean(self, item_id: int) -> float:
        """Returns the mean of the item"""
        return self.item_mean_ratings[item_id]