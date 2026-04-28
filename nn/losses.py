import numpy as np

def binary_cross_entropy(y, y_hat):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))