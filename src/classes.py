import numpy as np
import time

class Id_Converter:
    """Contains all relevant info about the element"""
    def __init__(self):    
        """Converts string ids to unique integer ids and stores relation in a dictionary"""
        self.user_ids = {}
        self.item_ids = {}

        """Global mean rating"""
        self.global_number_of_ratings = 0
        self.sum_of_all_ratings = 0

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

    def add_rating(self, rating: int) -> None:

        self.sum_of_all_ratings += rating
        self.global_number_of_ratings += 1

    def get_global_mean(self) -> float:
        return self.sum_of_all_ratings / self.global_number_of_ratings

