import numpy as np
import pandas as pd
from classes import Items, Id_Converter

def get_single_similarity(i: Items, u: int, v: int) -> float:
    """Calculates the pearson similarity between two items"""
    num, den1, den2 = 0, 0, 0
    # print("Item: ", v, "item ", u)
    for user_id, rating_u in i.get_pairs_for_item(u).items():
        # print("User: ", user_id, "rating: ", rating_u)
        if user_id in i.get_pairs_for_item(v):
            # print("User is in item ", v)
            num += (rating_u - i.get_mean(u)) * (i.get_rating(v, user_id) - i.get_mean(v))
            den1 += (rating_u - i.get_mean(u)) ** 2
            den2 += (i.get_rating(v, user_id) - i.get_mean(v)) ** 2
            print(num, den1, den2)

    if den1 == 0 or den2 == 0: return 0
    # print("Similaridade entre ", u, " e ", v, " = ", num / (np.sqrt(den1) * np.sqrt(den2)))
    return num / (np.sqrt(den1) * np.sqrt(den2))

def get_pearson_similarities_for_item(item_ratings: Items, user_id: int, item_id: int, n_items: int, k: int) -> dict:
    """Calculates the pearson similarity between all items"""
    item_ratings.compute_mean()

    i_similarities = []
    # print("Item: ", item_id, "User: ", user_id)
    #para cada usuario que usou o item da query vamos ver seus itens
    for c_id, rating in item_ratings.get_pairs_for_user(user_id).items():
        # print("Item: ", c_id, "Rating: ", rating)
        # print("vamos calcular a similaridade entre ", item_id, " e ", c_id)
        i_similarities.append((get_single_similarity(item_ratings, c_id, item_id), c_id))
        i_similarities.sort(key=lambda x: x[0], reverse = True)

    # print(i_similarities)
    return i_similarities[:k]

def predict_rating(item_ratings: Items, most_similar: list, user_id: int, item_id: int) -> float:
    """Predicts the rating of an item for a user"""
    num, den = 0, 0
    for similarity, c in most_similar: #[(similaridade(i, u), u), (similaridade(i, v), v), ...]
        if user_id in item_ratings.get_pairs_for_item(c):
            num += similarity * (item_ratings.get_rating(c, user_id) - item_ratings.get_mean(c))
            den += abs(similarity)
    if den == 0:
        return 0
    return item_ratings.get_mean(item_id) + num / den

def predict_all_targets(targets_df: pd.DataFrame, item_ratings: Items, id_converter: Id_Converter) -> list:
    """Predicts all targets in the dataframe"""
    similarities = {}
    predictions = []
    for row in targets_df.values:
        user_id = id_converter.convert_user(row[0]); item_id = id_converter.convert_item(row[1])  

        if item_id not in similarities:
            similarities[item_id] = get_pearson_similarities_for_item(item_ratings, user_id, item_id, id_converter.get_number_of_items(), 10)
        predictions.append(predict_rating(item_ratings, similarities[item_id], user_id, item_id)) 
    return predictions