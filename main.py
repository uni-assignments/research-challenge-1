import sys, time

# adding src folder to path
sys.path.insert(0, './src')

from data_parser import read_input_data, fill_ratings_matrix, get_rmse, output_predictions, read_as_dataframe
from classes import Id_Converter, Items
from item_based import predict_all_targets
from constants import *

if __name__ == '__main__': 

    complete_time = time.time()
    ratings_file, targets_file = sys.argv[1], sys.argv[2]
    
    start = time.time()
    id_converter, item_ratings = read_input_data(ratings_file)
    end = time.time()
    print(f"Tempo para criar r: {end - start} segundos")

    complete_time = time.time()
    target_df = read_as_dataframe(targets_file)
    predictions = predict_all_targets(target_df, item_ratings, id_converter)
    output_predictions(target_df, predictions)
    final_time = time.time()
    print(f"Tempo para tudo: {final_time - complete_time} segundos")


