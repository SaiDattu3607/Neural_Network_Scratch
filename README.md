# Neural Network from Scratch

A minimal implementation of a neural network from scratch using NumPy, trained on the MNIST dataset.

## Project Structure

```
neural_net_scratch/
├── nn/                    # Core neural network components
│   ├── layers.py         # Linear layer implementation
│   ├── activations.py    # Activation functions (ReLU, Softmax, Sigmoid)
│   └── losses.py         # Loss functions (Categorical Cross-Entropy)
├── models/               # Model architectures
│   └── mlp.py           # Multi-layer Perceptron (2-layer network)
├── data/                 # Data handling
│   ├── mnist_loader.py  # MNIST binary file loader
│   └── dataloader.py    # Batch DataLoader implementation
├── train/               # Training scripts
│   └── train_mnist.py   # MNIST training script
├── config.py            # Configuration and hyperparameters
├── inference.py         # Inference script for predictions
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Features

- **From Scratch**: Implemented using only NumPy, no deep learning frameworks
- **Forward & Backward Pass**: Full gradient computation for backpropagation
- **Flexible DataLoader**: Supports batching and shuffling
- **MNIST Support**: Direct binary file parsing for MNIST dataset

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the model on MNIST:

```bash
python train/train_mnist.py
```

This trains a 2-layer MLP with architecture:
- Input: 784 (28×28 MNIST images)
- Hidden: 128 neurons (ReLU activation)
- Output: 10 classes (Softmax activation)

### Configuration

Edit `config.py` to modify hyperparameters:
```python
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01
```

### Inference

Make predictions on new images:

```bash
python inference.py --image path/to/image.png
```

## Architecture

### Forward Pass
1. **Layer 1**: Linear(784 → 128) + ReLU
2. **Layer 2**: Linear(128 → 10) + Softmax

### Loss Function
Categorical Cross-Entropy with numerical stability

### Optimization
Stochastic Gradient Descent (SGD) with manual gradient updates

## Dataset

Download MNIST dataset from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/):

```bash
mkdir -p data/raw
# Download the 4 binary files into data/raw/
```

## Implementation Details

### Neural Network Components

**Linear Layer** (`nn/layers.py`):
- Forward: `z = Wx + b`
- Backward: Computes gradients `dW`, `db`, and `dX`

**Activation Functions** (`nn/activations.py`):
- **ReLU**: `max(0, x)` - Hidden layer
- **Softmax**: Normalized exponential - Output layer
- **Sigmoid**: `1/(1+e^-x)` - Optional

**Loss Function** (`nn/losses.py`):
- Categorical Cross-Entropy with clipping for numerical stability

## Performance

On MNIST dataset:
- Training: ~50,000 samples
- Testing: ~10,000 samples
- Expected accuracy: ~97% after 10 epochs

## License

MIT
