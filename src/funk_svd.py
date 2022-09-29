import numpy as np
import pandas as pd

from data_parser import get_rmse
from classes import Id_Converter

class FunkSVD:
    
    def __init__(self, ids_table: Id_Converter, init_mean: float, init_std: float):
        
        self.ids_table = ids_table
        self.init_mean = init_mean
        self.init_std = init_std
        self.mu = self.ids_table.get_global_mean()
        

    def init_matrix(self, n_factors: int) -> (np.ndarray, np.ndarray):
        
        """Initialize the user and item matrixes"""
        np.random.seed(1)

        number_of_user = self.ids_table.get_number_of_users()
        number_of_items = self.ids_table.get_number_of_items()

        self.user_matrix = np.random.normal(loc = self.init_mean, scale = self.init_std, size = (number_of_user, n_factors)) 
        self.item_matrix = np.random.normal(loc = self.init_mean, scale = self.init_std, size = (number_of_items, n_factors)) 
        self.user_bias = np.zeros(number_of_user)
        self.item_bias = np.zeros(number_of_items)


    def sgd(self, train_data, validation_data, n_factors = 20, learning_rate = 0.005, reg_param = 0.02, n_epochs = 20, epsilon = 0.001                              ,
            early_stop = True, verbose = True) -> (np.ndarray, np.ndarray):
        
        """Finds values for P and Q matrixes such that R = P x Q using SGD"""
        epochs_rmse = [0] * n_epochs
        self.init_matrix(n_factors)
        for epoch in range(n_epochs):
            for (user, item), rating in train_data.items():

                pred = self.mu + self.user_bias[user] + self.item_bias[item] + np.dot(self.user_matrix[user, :], self.item_matrix[item, :].T)
                error = float(rating) - pred
                
                user_aux = self.user_matrix[user, :] + learning_rate * (error * self.item_matrix[item, :] - reg_param * self.user_matrix[user,:])
                item_aux = self.item_matrix[item, :] + learning_rate * (error * self.user_matrix[user, :] - reg_param * self.item_matrix[item,:])
                user_bias_aux = self.user_bias[user] + learning_rate * (error - reg_param * self.user_bias[user])
                item_bias_aux = self.item_bias[item] + learning_rate * (error - reg_param * self.item_bias[item])

                self.user_bias[user] = user_bias_aux
                self.item_bias[item] = item_bias_aux                
                self.user_matrix[user, :] = user_aux
                self.item_matrix[item, :] = item_aux
            
            if validation_data is not None:
                predictions, true_values = self.predict_test_cases(validation_data)
                epochs_rmse[epoch] = get_rmse(predictions, true_values)
                if verbose:
                    print(f"Epoch {epoch}: RMSE = {epochs_rmse[epoch]}")

                if epoch > 0 and early_stop:
                    if epochs_rmse[epoch - 1] - epochs_rmse[epoch] < epsilon:
                        if verbose:
                            print(f"Early stop at epoch {epoch}, because RMSE is increasing")
                        break

    def predict(self, user_id: int, item_id: int) -> float:
        
        """Predicts the rating of a user for an item"""
        pred = np.dot(self.user_matrix[user_id, :], self.item_matrix[item_id, :].T) + self.mu + self.user_bias[user_id] + self.item_bias[item_id]
        if pred > 5:
            return 5
        elif pred < 1:
            return 1
        else:
            return pred

    def predict_test_cases(self, targets: dict) -> (np.ndarray, np.ndarray):
        
        """Predicts all targets in the dataframe"""
        predictions = []
        true_values = []

        for (user, item), rating in targets.items():    
            predictions.append(self.predict(user, item))
            true_values.append(rating)
        
        return np.array(predictions, dtype=np.float32), np.array(true_values, dtype=np.float32)

    def predict_all_targets(self, targets_df: pd.DataFrame) -> np.ndarray:

        """Predicts all targets in the dataframe"""
        predictions = []
        for row in targets_df.values:
            user = self.ids_table.convert_user(row[0])
            item = self.ids_table.convert_item(row[1])
            predictions.append(self.predict(user, item))

        return np.array(predictions, dtype=np.float32)
