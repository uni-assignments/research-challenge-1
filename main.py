import sys, time

# adding src folder to path
sys.path.insert(0, './src')

from data_parser import read_as_dataframe, read_r_matrix, get_rmse
from funk_svd import sgd, predict_all_targets
from constants import *

if __name__ == '__main__': 
    ratings_file, targets_file = sys.argv[1], sys.argv[2]
    
    start = time.time()
    r, user_ids, item_ids = read_r_matrix(ratings_file)
    end = time.time()
    print(f"Tempo para criar r: {end - start} segundos")

    start = time.time()
    user_matrix, item_matrix = sgd(r, user_ids.__no_elements__(), item_ids.__no_elements__())
    end = time.time()
    print(f"Tempo para gradiente descendente: {end - start} segundos")
    
    targets_df = read_as_dataframe(test_file)
    predictions = predict_all_targets(targets_df, user_matrix, item_matrix, user_ids, item_ids)
    rmse =  get_rmse(predictions, targets_df['Rating'])
    print(f"RMSE: {rmse}")
    # output_predictions(targets_df, predictions)



