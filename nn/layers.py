"""Linear layer implementation for neural networks."""

import numpy as np


class Linear:
    """Fully connected (Dense) layer.
    
    Implements a linear transformation: z = Wx + b
    """
    
    def __init__(self, in_features, out_features):
        """Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros((out_features, 1))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        """Forward pass: z = Wx + b.
        
        Args:
            x: Input activation of shape (in_features, batch_size)
            
        Returns:
            Output of shape (out_features, batch_size)
        """
        self.x = x
        return np.dot(self.W, x) + self.b
    
    def backward(self, grad_output):
        """Backward pass: compute gradients.
        
        Args:
            grad_output: Gradient of loss w.r.t output (dL/dz)
            
        Returns:
            Gradient of loss w.r.t input (dL/dx)
        """
        self.dW = np.dot(grad_output, self.x.T)  # dL/dW
        self.db = grad_output  # dL/db
        return np.dot(self.W.T, grad_output)  # dL/dx