import sys, time
import pandas as pd
import numpy as np
# adding src folder to path
sys.path.insert(0, './src')

from data_parser import read_as_dataframe, read_input_data, split_train_test_val, get_rmse, output_predictions
from classes import Id_Converter
from funk_svd import FunkSVD
from constants import *

def first_model(target_df: pd.DataFrame, ids_table: Id_Converter, train: dict, val: dict, teste: dict, experimental = True) -> np.ndarray:

    print(f"Treinando modelo com inicializacao : {0.0} e {0.20} e {7} fatores latentes")

    model = FunkSVD(ids_table, init_mean = 0.0, init_std = 0.20)
    model.sgd(train_data = train, validation_data = val, n_factors = 7, learning_rate = 0.005, reg_param = 0.03, 
        n_epochs = 22, epsilon = 0.0001, early_stop = True, verbose = True)

    if experimental:
        predictions, true_values = model.predict_test_cases(test)
        rmse = get_rmse(predictions, true_values)
        print(f"RMSE no conjunto de teste: {rmse}")

    return model.predict_all_targets(target_df)

def second_model(target_df: pd.DataFrame, ids_table: Id_Converter, train: dict, val: dict, test: dict, experimental = True) -> np.ndarray:

    print(f"Treinando modelo com inicializacao : {0.0} e {0.16} e {5} fatores latentes")

    model = FunkSVD(ids_table, init_mean = 0.0, init_std = 0.16)
    model.sgd(train_data = train, validation_data = val, n_factors = 5, learning_rate = 0.005, reg_param = 0.03, 
        n_epochs = 22, epsilon = 0.0001, early_stop = True, verbose = True)

    if experimental:
        predictions, true_values = model.predict_test_cases(test)
        rmse = get_rmse(predictions, true_values)
        print(f"RMSE no conjunto de teste: {rmse}")

    return model.predict_all_targets(target_df)

if __name__ == '__main__': 

    complete_time = time.time()
    ratings_file, targets_file = sys.argv[1], sys.argv[2]
    
    # Leitura de Dados
    start = time.time()
    input_data, ids_table = read_input_data(ratings_file)
    end = time.time()
    print(f"Tempo para criar r: {end - start} segundos")


    #Divisão de Dados para treinos experimentais
    # start = time.time()
    # train, val, test = split_train_test_val(input_data, test_size = 0.2, val_size = 0.2)
    # end = time.time()
    # print(f"Tempo para dividir conjunto de dados: {end - start} segundos")

    target_df = read_as_dataframe(targets_file)

    #Primeiro Modelo
    start = time.time()
    pred1 = first_model(target_df, ids_table, input_data, None, None,  experimental = False)
    end = time.time()
    print(f"Tempo para rodar primeiro modelo: {end - start} segundos")

    #Segundo Modelo
    start = time.time()
    pred2 = second_model(target_df, ids_table,  input_data, None, None, experimental = False)
    end = time.time()
    print(f"Tempo para rodar segundo modelo: {end - start} segundos")
    
    output_predictions(target_df, (pred1 + pred2)/2)

    final_time = time.time()
    print(f"Tempo para tudo: {final_time - complete_time} segundos")

