"""Multi-Layer Perceptron model."""

from nn.layers import Linear
from nn.activations import sigmoid, relu, softmax


class MLP:
    """2-layer Multi-Layer Perceptron for MNIST classification.
    
    Architecture:
    - Input: 784 features (28×28 images flattened)
    - Hidden: 128 neurons with ReLU activation
    - Output: 10 neurons with Softmax activation
    """
    
    def __init__(self):
        """Initialize MLP with random weights."""
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)
        
        # Cache intermediate activations for backward pass
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input of shape (784, batch_size)
            
        Returns:
            Output probabilities of shape (10, batch_size)
        """
        # Layer 1: Linear + ReLU
        self.z1 = self.l1.forward(x)
        self.a1 = relu(self.z1)

        # Layer 2: Linear + Softmax
        self.z2 = self.l2.forward(self.a1)
        self.a2 = softmax(self.z2)

        return self.a2

