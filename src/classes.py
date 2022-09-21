import numpy as np
import time

class Table:
    """Contains all relevant info about the element"""
    def __init__(self):    
        """Converts string ids to unique integer ids and stores relation in a dictionary"""
        self.id_converter = {}
        """Stores the list of ratings relative to the element"""
        self.ratings = {}
        """Stores the mean of the ratings relative to the element"""
        self.means = {}

    def size(self):
        return len(self.id_converter)
    
    def element_exists(self, id: str):
        return id in self.id_converter

    def add_id(self, old_id: str) -> int:
        """Checks if the id is already in the dictionary, if not adds it and returns the new id"""
        if not self.element_exists(old_id):
            self.id_converter[old_id] = len(self.id_converter)
        return self.id_converter[old_id]

    def convert(self, old_id: str) -> int:
        """Converts the string id to the integer id"""
        return self.id_converter[old_id]

    def add_rating(self, id: int, rating: int):
        if id in self.ratings:
            self.ratings[id].append(rating)
        else:
            self.ratings[id] = [rating]

    def get_rating(self, id: int):
        return self.ratings[id]

    def compute_mean(self):
        for id, ratings in self.ratings.items():
            self.means[id] = sum(ratings) / len(ratings)
            
    def get_mean(self, id: int):
        return self.means[id]


class User_Table (Table):
  def __init__(self):
    super().__init__()

class Item_Table (Table):
  def __init__(self):
    super().__init__()