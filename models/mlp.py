from nn.layers import Linear
from nn.activations import sigmoid
from nn.activations import relu
from nn.activations import softmax

class MLP:
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def forward(self, x):
        self.z1 = self.l1.forward(x)
        self.a1 = relu(self.z1)

        self.z2 = self.l2.forward(self.a1)
        self.a2 = softmax(self.z2)

        return self.a2

