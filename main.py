import sys, time

# adding src folder to path
sys.path.insert(0, './src')

from data_parser import read_as_dataframe, read_r_matrix, split_train_test_val, get_rmse
from funk_svd import FunkSVD
from constants import *

if __name__ == '__main__': 

    ratings_file, targets_file = sys.argv[1], sys.argv[2]
    
    start = time.time()
    input_data, user_ids, item_ids = read_r_matrix(ratings_file)
    end = time.time()
    print(f"Tempo para criar r: {end - start} segundos")

    start = time.time()
    train, test, val = split_train_test_val(input_data)
    end = time.time()
    print(f"Tempo para dividir conjunto de dados: {end - start} segundos")

    start = time.time()
    model = FunkSVD(user_ids.__no_elements__(), item_ids.__no_elements__())
    model.sgd(train, val, n_factors = 15, learning_rate = 0.0005, reg_param = 0.02, n_epochs = 50, early_stop = True, verbose = True)
    end = time.time()
    print(f"Tempo para rodar modelo: {end - start} segundos")
    
    predictions, true_values = model.predict_all_targets(test)
    rmse = get_rmse(predictions, true_values)
    print(f"RMSE no conjunto de teste: {rmse}")
    # output_predictions(targets_df, predictions)



