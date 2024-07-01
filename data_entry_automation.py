# data_entry_automation.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import MODEL_PATH, IMAGES_DIR
import numpy as np
import os


def automate_data_entry(images_directory, model_path):
    # Load the trained model
    model = load_model(model_path)

    # Get a list of all image files in the directory
    image_files = os.listdir(images_directory)

    # Initialize an empty list to store the digit predictions
    digit_predictions = []

    for image_file in image_files:
        # Load the image
        image = load_img(os.path.join(images_directory, image_file), color_mode='grayscale', target_size=(28, 28))

        # Convert the image to a numpy array and normalize it
        image = img_to_array(image) / 255.0

        # Reshape the image array for the model
        image = np.expand_dims(image, axis=0)

        # Predict the digit and add it to the list of predictions
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        digit_predictions.append(digit)

    return digit_predictions


# Usage example
images_dir = IMAGES_DIR
trained_model_path = MODEL_PATH
predictions = automate_data_entry(images_dir, trained_model_path)

# Print each predicted digit on a new line
for i, predicted_digit in enumerate(predictions):
    print(f'The predicted digit for image {i + 1} is: {predicted_digit}')
