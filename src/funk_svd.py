import numpy as np
import pandas as pd

from data_parser import Id_Converter, get_rmse

class FunkSVD:
    def __init__(self, no_users: int, no_items: int):
        
        self.no_users = no_users
        self.no_items = no_items


    def init_matrix(self, n_factors: int) -> (np.ndarray, np.ndarray):
        
        """Initialize the user and item matrixes"""
        self.user_matrix = np.random.rand(self.no_users, n_factors)
        self.item_matrix = np.random.rand(self.no_items, n_factors)
        

    def sgd(self, train_data, validation_data, n_factors = 15, 
            learning_rate = 0.0005, reg_param = 0.02, n_epochs = 50, early_stop = True, verbose = True) -> (np.ndarray, np.ndarray):
        
        """Finds values for P and Q matrixes such that R = P x Q using SGD"""
        epochs_rmse = [0] * n_epochs
        self.init_matrix(n_factors)
        for epoch in range(n_epochs):
            for (user, item), rating in train_data.items():
                pred = np.dot(self.user_matrix[user, :], self.item_matrix[item, :].T)
                error = float(rating) - pred
                user_aux = self.user_matrix[user, :] + learning_rate * (error * self.item_matrix[item, :] - reg_param * self.user_matrix[user,:])
                item_aux = self.item_matrix[item, :] + learning_rate * (error * self.user_matrix[user, :] - reg_param * self.item_matrix[item,:])
                self.user_matrix[user, :] = user_aux
                self.item_matrix[item, :] = item_aux
            
            predictions, true_values = self.predict_all_targets(validation_data)
            epochs_rmse[epoch] = get_rmse(predictions, true_values)
            if verbose:
                print(f"Epoch {epoch}: RMSE = {epochs_rmse[epoch]}")

            if epoch > 0 and early_stop:
                if epochs_rmse[epoch] > epochs_rmse[epoch - 1]:
                    if verbose:
                        print(f"Early stop at epoch {epoch}, because RMSE is increasing")
                    break

    def predict(self, user_id: int, item_id: int) -> float:
        
        """Predicts the rating of a user for an item"""
        return np.dot(self.user_matrix[user_id, :], self.item_matrix[item_id, :].T)


    def predict_all_targets(self, targets: dict) -> (np.ndarray, np.ndarray):
        
        """Predicts all targets in the dataframe"""
        predictions = []
        true_values = []

        for (user, item), rating in targets.items():    
            predictions.append(self.predict(user, item))
            true_values.append(rating)
        
        return np.array(predictions, dtype=np.float32), np.array(true_values, dtype=np.float32)