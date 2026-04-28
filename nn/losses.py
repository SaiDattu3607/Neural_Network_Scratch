import numpy as np

def categorical_cross_entropy(y, y_hat):
    eps = 1e-8
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.sum(y * np.log(y_hat)) / y.shape[1]