import sys, time

# adding src folder to path
sys.path.insert(0, './src')

from data_parser import read_as_dataframe, append_integer_ids_to_df, generate_r_matrix, read_as_matrix
from funk_svd import init_matrixes, sgd

if __name__ == '__main__': 
    ratings_file, targets_file = sys.argv[1], sys.argv[2]
    start = time.time()

    # ratings_df = read_as_dataframe(ratings_file)
    # ratings_df, user_ids, item_ids = append_integer_ids_to_df(ratings_df)
    # r = generate_r_matrix(ratings_df, user_ids, item_ids)
    r, user_ids, item_ids = read_as_matrix(ratings_file)
    end = time.time()
    print(f"Tempo para criar r: {end - start} segundos")

    # start = time.time()
    # user_matrix, item_matrix = sgd(r, user_ids.__no_elements__(), item_ids.__no_elements__())
    # end = time.time()
    # print(f"Tempo para gradiente descendente: {end - start} segundos")
    # targets_df = read_as_dataframe(targets)
    # predictions = predict(targets_df)
    # output_predictions(targets_df, predictions)



