import numpy as np

def init_matrixes(n_users: int, n_items: int, n_factors: int) -> (np.ndarray, np.ndarray):
    """Initialize the user and item matrixes"""
    user_matrix = np.random.rand(n_users, n_factors)
    item_matrix = np.random.rand(n_items, n_factors)
    
    return user_matrix, item_matrix

def sgd(r: np.ndarray, n_users: int, n_items: int, n_factors = 100, 
    learning_rate = 0.005, reg_param = 0.02, n_epochs = 20) -> (np.ndarray, np.ndarray):
    """Perform matrix factorization using SGD"""

    user_matrix, item_matrix = init_matrixes(n_users, n_items, n_factors)
    for epoch in range(n_epochs):
        for (user, item), rating in r.items():
            error = rating - np.dot(user_matrix[user, :], item_matrix[item, :].T)
            user_aux = user_matrix[user, :] + learning_rate * (error * item_matrix[item, :] - reg_param * user_matrix[user,:])
            item_aux = item_matrix[item, :] + learning_rate * (error * user_matrix[user, :] - reg_param * item_matrix[item,:])
            user_matrix[user, :] = user_aux
            item_matrix[item, :] = item_aux

    return user_matrix, item_matrix


def predict(user_matrix: np.ndarray, item_matrix: np.ndarray, user_id: int, item_id: int):
    pass