import pandas as pd
import numpy as np

from classes import User_Table, Item_Table

def read_input_data(file_path: str) -> (np.ndarray, User_Table, Item_Table):
    """Reads a file and returns a r matrix"""
    
    r = {}
    user_table = User_Table()
    item_table = Item_Table()

    users_ratings = {}
    items_ratings = {}

    f = open(file_path, "r"); next(f)
    for line in f:
        line = line.replace('\n', '').replace(':', ',').split(',')
        user_int_id = user_table.add_id(line[0])
        item_int_id = item_table.add_id(line[1])
        r[(user_int_id, item_int_id)] = int(line[2])

        user_table.add_rating(user_int_id, int(line[2]))
        item_table.add_rating(item_int_id, int(line[2]))
    f.close()
    
    return r, user_table, item_table

def weighted_avarage(input_data: np.ndarray, user_table: User_Table, item_table: Item_Table) -> np.ndarray:
    
    """Calculates the weighted average of the ratings matrix"""
    for (user, item), rating in input_data.items():
        input_data[(user, item)] -= user_table.get_mean(user)
    return input_data

def read_as_dataframe(file_path: str) -> pd.DataFrame:
    """Reads a file and returns a pandas dataframe"""
    return pd.read_csv(file_path, sep=r';|,|:', engine='python')

def split_train_test_val(input_data: dict, test_size: float = 0.1, val_size: float = 0.1) -> (np.ndarray, np.ndarray):
            
    """Splits the ratings matrix into train, test and validation"""
    test = {}
    train = {}
    val = {}
    for (user, item), rating in input_data.items():
        if np.random.rand() < test_size:
            test[(user, item)] = rating
        elif np.random.rand() < val_size + test_size:
            val[(user, item)] = rating
        else:
            train[(user, item)] = rating
    
    return train, test, val

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