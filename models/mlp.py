from nn.layers import Linear
from nn.activations import sigmoid

class MLP:
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def forward(self, x):
        z1 = self.l1.forward(x)
        a1 = sigmoid(z1)

        z2 = self.l2.forward(a1)
        a2 = sigmoid(z2)

        return a2