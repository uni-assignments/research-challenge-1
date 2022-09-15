import pandas as pd
import numpy as np
import time

def read_as_dataframe(file_path: str) -> pd.DataFrame:
    """Reads a file and returns a pandas dataframe"""
    df = pd.read_csv(file_path, sep=r';|,|:', engine='python')
    return df

class Id_Converter:
    """Converts string ids to unique integer ids and stores relation in a dictionary"""
    def __init__(self):
        self.dict = {}

    def __create_new_id__(self, old_id: str) -> int:
        if old_id not in self.dict:
            self.dict[old_id] = len(self.dict)
        return self.dict[old_id]

    def __convert__(self, old_id: str) -> int:
        return self.dict[old_id]

    def __no_elements__(self):
        return len(self.dict)


def read_r_matrix(file_path: str) -> (np.ndarray, Id_Converter, Id_Converter):
    """Reads a file and returns a r matrix"""
    
    r = {}
    user_ids = Id_Converter()
    item_ids = Id_Converter()

    f = open(file_path, "r")
    next(f)
    for line in f:
        line = line.replace('\n', '').replace(':', ',').split(',')
        user_int_id = user_ids.__create_new_id__(line[0])
        item_int_id = item_ids.__create_new_id__(line[1])
        r[(user_int_id, item_int_id)] = line[2]
    f.close()
    
    return r, user_ids, item_ids

def get_rmse(predictions, targets):
    """Calculates the root mean squared error between predictions and targets"""
    return np.sqrt(((predictions - targets) ** 2).mean())



def append_integer_ids_to_df(df: pd.DataFrame) -> (pd.DataFrame, Id_Converter, Id_Converter):
    """Associates user and item string ids to integer ones and adds to the dataframe"""
    # start = time.time()

    user_ids = Id_Converter()
    item_ids = Id_Converter()

    user_list, item_list = [], []
    for row in df.values:  
        user_list.append(user_ids.__create_new_id__(row[0]))
        item_list.append(item_ids.__create_new_id__(row[1]))

    df['IntUserId'], df['IntItemId']  = user_list, item_list
    
    # end = time.time()
    # print(f"Tempo para adicionar ids: {end - start} segundos")
    return df, user_ids, item_ids

def generate_r_matrix(df, user_ids, item_ids):
    # start = time.time()

    r = {}
    for row in df.values:  
        user_int_id = user_ids.__convert__(row[0])
        item_int_id = item_ids.__convert__(row[1])
        r[(user_int_id, item_int_id)] = row[2]

    # end = time.time()
    # print(f"Tempo para adicionar ids: {end - start} segundos")
    return r

def output_predictions(target_df: pd.DataFrame, predictions_matrix: np.ndarray) -> None:
    """Outputs the predictions to the standard output"""
    pass