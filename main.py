import sys, time

# adding src folder to path
sys.path.insert(0, './src')

from data_parser import read_as_dataframe, read_input_data, weighted_avarage, split_train_test_val, get_rmse, output_predictions
from classes import User_Table, Item_Table
from funk_svd import FunkSVD
from constants import *

if __name__ == '__main__': 

    complete_time = time.time()
    ratings_file, targets_file = sys.argv[1], sys.argv[2]
    
    start = time.time()
    input_data, user_table, item_table = read_input_data(ratings_file)
    end = time.time()
    print(f"Tempo para criar r: {end - start} segundos")

    start = time.time()
    train, test, val = split_train_test_val(input_data)
    end = time.time()
    print(f"Tempo para dividir conjunto de dados: {end - start} segundos")

    start = time.time()
    model = FunkSVD(user_table, item_table)
    model.sgd(train, val, n_factors = 17, learning_rate = 0.005, reg_param = 0.01, n_epochs = 100, early_stop = True, verbose = True)
    end = time.time()
    print(f"Tempo para rodar modelo: {end - start} segundos")
    
    predictions, true_values = model.predict_test_cases(test)
    rmse = get_rmse(predictions, true_values)
    print(f"RMSE no conjunto de teste: {rmse}")

    target_df = read_as_dataframe(targets_file)
    predictions = model.predict_all_targets(target_df)
    output_predictions(target_df, predictions)

    final_time = time.time()
    print(f"Tempo para tudo: {final_time - complete_time} segundos")


