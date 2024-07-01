# config.py

import os
from datetime import datetime

# Directory Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
IMAGES_DIR = 'images'

# CSV File Names
TRAIN_CSV_NAME = 'train.csv'
TEST_CSV_NAME = 'test.csv'
CLEANED_CSV_NAME = 'cleaned.csv'

# Full Paths
TRAIN_CSV_PATH = os.path.join(DATA_DIR, TRAIN_CSV_NAME)
TEST_CSV_PATH = os.path.join(DATA_DIR, TEST_CSV_NAME)
CLEANED_CSV_PATH = os.path.join(DATA_DIR, CLEANED_CSV_NAME)

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model Saving Configuration
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
MODEL_NAME = 'mnist_neural_network_model_20240630192221.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Training Hyperparameters
EPOCHS = 100
VALIDATION_SPLIT = 0.3
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Model Configuration
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']
EARLY_STOPPING_MONITOR = 'val_loss'
DENSE_LAYER_ACTIVATION = 'relu'
OUTPUT_LAYER_ACTIVATION = 'softmax'
DROPOUT_RATE = 0.3
OPTIMIZER = 'adam'
OUTPUT_LAYER_UNITS = 10
INPUT_SHAPE = (784,)
