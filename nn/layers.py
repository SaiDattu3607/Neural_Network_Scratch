import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros((out_features, 1))

    def forward(self, x): # x is activation from previous layer
        self.x = x
        return np.dot(self.W, x) + self.b
    
    def backward(self, grad_output):
        self.dW = np.dot(grad_output, self.x.T) # dL/dW
        self.db = grad_output # dL/db
        return np.dot(self.W.T, grad_output) # dL/dx