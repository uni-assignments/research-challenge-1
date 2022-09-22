import pandas as pd
import numpy as np

from classes import Id_Converter, Items

def read_as_dataframe(file_path: str) -> pd.DataFrame:
    """Reads a file and returns a pandas dataframe"""
    return pd.read_csv(file_path, sep=r';|,|:', engine='python')

def read_input_data(file_path: str) -> (np.ndarray, Id_Converter, Items):
    """Reads a file and returns a r matrix"""

    # r = {}
    id_converter = Id_Converter()

    item_ratings = Items()

    f = open(file_path, "r"); next(f)
    for line in f:
        line = line.replace('\n', '').replace(':', ',').split(',')
        user_int_id = id_converter.add_id(type='User', old_id=line[0])
        item_int_id = id_converter.add_id(type='item', old_id=line[1])

        item_ratings.add(item_int_id, user_int_id, float(line[2]))
        # r[(user_int_id, item_int_id)] = int(line[2])

    f.close()
    
    # return r, id_converter, item_ratings
    return id_converter, item_ratings

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

def fill_ratings_matrix(r: np.ndarray, id_converter: Id_Converter) -> np.ndarray:
    r = np.zeros([id_converter.get_number_of_items(), id_converter.get_number_of_users], dtype = np.float32)
    """Fills the ratings matrix with the values from the dictionary"""
    for (user_id, item_id), rating in r.items():
        r[user_i][item_id] = rating
    return r

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