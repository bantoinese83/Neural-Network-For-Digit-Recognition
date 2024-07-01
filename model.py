from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from config import DENSE_LAYER_ACTIVATION, OUTPUT_LAYER_ACTIVATION, DROPOUT_RATE, OUTPUT_LAYER_UNITS, \
    OPTIMIZER, LOSS, METRICS


def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(512, activation=DENSE_LAYER_ACTIVATION, kernel_regularizer=l2(0.01)),
        Dropout(DROPOUT_RATE),
        Dense(512, activation=DENSE_LAYER_ACTIVATION, kernel_regularizer=l2(0.01)),
        Dropout(DROPOUT_RATE),
        Dense(OUTPUT_LAYER_UNITS, activation=OUTPUT_LAYER_ACTIVATION)
    ])
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    return model
