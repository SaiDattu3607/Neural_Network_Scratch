# Training Configuration

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01

# Model Architecture
INPUT_SIZE = 784  # 28 x 28 MNIST images
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10  # 10 digit classes

# Data
DATA_DIR = "data/raw"
TRAIN_SIZE = 60000
TEST_SIZE = 10000

# Logging
VERBOSE = True
SAVE_CHECKPOINTS = False
CHECKPOINT_DIR = "checkpoints"
