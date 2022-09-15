import numpy as np
import pandas as pd

from data_parser import Id_Converter

def init_matrix(n_users: int, n_items: int, n_factors: int) -> (np.ndarray, np.ndarray):
    
    """Initialize the user and item matrixes"""
    user_matrix = np.random.rand(n_users, n_factors)
    item_matrix = np.random.rand(n_items, n_factors)
    
    return user_matrix, item_matrix

def split_train_test_val(r: np.ndarray, test_size: float = 0.1, val_size: float = 0.1) -> (np.ndarray, np.ndarray):
        
        """Splits the ratings matrix into train, test and validation"""
        test = {}
        train = {}
        val = {}
        for (user, item), rating in r.items():
            if np.random.rand() < test_size:
                test[(user, item)] = rating
            elif np.random.rand() < val_size + test_size:
                val[(user, item)] = rating
            else:
                train[(user, item)] = rating
        
        return train, test, val


def sgd(r: np.ndarray, n_users: int, n_items: int, n_factors = 10, 
    learning_rate = 0.001, reg_param = 0.02, n_epochs = 50, early_stop = True) -> (np.ndarray, np.ndarray):
    
    """Finds values for P and Q matrixes such that R = P x Q using SGD"""
    user_matrix, item_matrix = init_matrix(n_users, n_items, n_factors)
    for epoch in range(n_epochs):
        for (user, item), rating in r.items():
            pred = np.dot(user_matrix[user, :], item_matrix[item, :].T)
            error = float(rating) - pred
            user_aux = user_matrix[user, :] + learning_rate * (error * item_matrix[item, :] - reg_param * user_matrix[user,:])
            item_aux = item_matrix[item, :] + learning_rate * (error * user_matrix[user, :] - reg_param * item_matrix[item,:])
            user_matrix[user, :] = user_aux
            item_matrix[item, :] = item_auxs

    return user_matrix, item_matrix


def predict(user_matrix: np.ndarray, item_matrix: np.ndarray, user_id: int, item_id: int) -> float:
    
    """Predicts the rating of a user for an item"""
    return np.dot(user_matrix[user_id, :], item_matrix[item_id, :].T)


def predict_all_targets(targets_df: pd.DataFrame, user_matrix: np.ndarray, item_matrix: np.ndarray, 
    user_ids: Id_Converter, item_ids: Id_Converter) -> np.ndarray:
    
    """Predicts all targets in the dataframe"""
    predictions = []
    for row in targets_df.values:
        user_id = user_ids.__convert__(row[0])
        item_id = item_ids.__convert__(row[1])
        predictions.append(predict(user_matrix, item_matrix, user_id, item_id))
    
    return np.array(predictions)