# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loguru import logger

from config import EPOCHS, BATCH_SIZE, PATIENCE, VALIDATION_SPLIT, LOSS, METRICS, EARLY_STOPPING_MONITOR, \
    OPTIMIZER
from model import create_model
from preprocessing import preprocess_data


def learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9


def train_model(df):
    # Check if df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.error("Input should be a pandas DataFrame")
        return None, None

    # Preprocess the data
    X, y = preprocess_data(df)

    # Check if the DataFrame is not empty
    if X.empty or y.empty:
        logger.error("DataFrame should not be empty")
        return None, None

    # Create the model
    model = create_model()

    # Log model summary
    model.summary(print_fn=logger.info)

    # Check if model is a Keras Model
    if not isinstance(model, Model):
        logger.error("Model should be a Keras Model")
        return None, None

    # Compile the model with the specified learning rate
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    # Create an ImageDataGenerator object for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    )

    # Convert DataFrame to a numpy array and then reshape
    X = X.values.reshape(-1, 28, 28, 1)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)

    # Fit the ImageDataGenerator object to your training data
    datagen.fit(X_train)

    # Create an EarlyStopping callback
    early_stopping = EarlyStopping(monitor=EARLY_STOPPING_MONITOR, patience=PATIENCE, verbose=1)

    # Create a LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)

    # Train the model
    try:
        history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                            validation_data=(X_val, y_val),
                            epochs=EPOCHS,
                            callbacks=[early_stopping, lr_scheduler
                                       ],
                            verbose=1)

    except Exception as e:
        logger.error(f"Failed to train the model. Error: {str(e)}")
        return None, None

    # Log training and validation loss/accuracy per epoch
    for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
        if metric in history.history:
            logger.info(f"{metric}: {history.history[metric]}")

    logger.info("Model trained successfully")
    return model, history
