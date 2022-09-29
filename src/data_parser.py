import pandas as pd
import numpy as np

from classes import Id_Converter

def read_input_data(file_path: str) -> (np.ndarray, Id_Converter):
    """Reads a file and returns a r matrix"""
    
    r = {}
    ids_table = Id_Converter()
    
    f = open(file_path, "r"); next(f)
    for line in f:
        line = line.replace('\n', '').replace(':', ',').split(',')
        user_int_id = ids_table.add_id(type='User', old_id=line[0])
        item_int_id = ids_table.add_id(type='item', old_id=line[1])
        
        ids_table.add_rating(int(line[2]))
        r[(user_int_id, item_int_id)] = int(line[2])
    
    f.close()
    
    return r, ids_table

def read_as_dataframe(file_path: str) -> pd.DataFrame:
    """Reads a file and returns a pandas dataframe"""
    return pd.read_csv(file_path, sep=r';|,|:', engine='python')

def split_train_test_val(input_data: dict, test_size: float = 0.2, val_size: float = 0.2) -> (np.ndarray, np.ndarray):
            
    """Splits the ratings matrix into train, test and validation"""
    test = {}
    train = {}
    val = {}

    train_size = 1 - test_size - val_size

    for row_idx, ((user, item), rating) in enumerate(input_data.items()):

        if row_idx < (len(input_data) * train_size):
            train[(user, item)] = rating        
        elif len(input_data) * train_size < row_idx and row_idx < len(input_data) * (train_size + val_size):
            val[(user, item)] = rating
        else:
            test[(user, item)] = rating
   
    return train, val, test

def get_rmse(predictions: np.ndarray, true_values: np.ndarray) -> float:

    """Calculates the root mean squared error between the predictions and the targets"""
    return np.sqrt(((predictions - true_values) ** 2).mean())

def output_predictions(target_df: pd.DataFrame, predictions: np.ndarray) -> None:
    
    """Outputs the predictions to a file"""
    f = open("output.csv", "a")
    f.write("UserId:ItemId,Rating\n")
    for idx, row in enumerate(target_df.values):
        f.write("{}:{},{:.4f}\n".format(row[0], row[1], predictions[idx]))
    f.close()