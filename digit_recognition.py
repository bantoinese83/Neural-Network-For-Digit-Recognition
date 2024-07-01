# digit_recognition.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import MODEL_PATH, IMAGES_DIR
import numpy as np
import os


def recognize_digit(image_path, model):
    # Load the image
    image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))

    # Convert the image to a numpy array and normalize it
    image = img_to_array(image) / 255.0

    # Reshape the image array for the model
    image = np.expand_dims(image, axis=0)

    # Predict the digit
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    return digit


# Usage example
images_dir = IMAGES_DIR
trained_model_path = MODEL_PATH

# Load the trained model
trained_model = load_model(trained_model_path)

predictions = []

# Get a list of all image files in the directory
image_files = os.listdir(images_dir)

# Loop over each image file
for image_file in image_files:
    # Construct the full image path
    digits_image_path = os.path.join(images_dir, image_file)

    # Use the recognize_digit function to predict the digit
    predicted_digit = recognize_digit(digits_image_path, trained_model)

    # Append the predicted digit to the prediction list
    predictions.append(predicted_digit)

# Print each predicted digit on a new line
for i, predicted_digit in enumerate(predictions):
    print(f'The predicted digit for image {i + 1} is: {predicted_digit}')
