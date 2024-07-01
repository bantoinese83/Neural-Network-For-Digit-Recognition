# Neural Network from Scratch

This project is a Python implementation of a neural network for digit recognition. The neural network is trained on the MNIST dataset.

## Project Structure

The project consists of several Python scripts and modules:

- `data_load.py`: Contains the `DataLoader` class for loading and preprocessing the data.
- `data_visualization.py`: Contains functions for visualizing the data.
- `preprocessing.py`: Contains functions for preprocessing the data.
- `model.py`: Contains the function `create_model` for creating the neural network model.
- `train.py`: Contains the function `train_model` for training the model.
- `evaluate.py`: Contains the function `evaluate_model` for evaluating the model.
- `utils.py`: Contains utility functions for loading and saving data and models.
- `config.py`: Contains configuration variables for the project.
- `main.py`: The main script that ties everything together.
- `digit_recognition.py`: Script for recognizing digits from images using the trained model.
- `data_entry_automation.py`: Script for automating data entry using the trained model.

## Usage

1. Run `main.py` to train the model and evaluate it.
2. Run `digit_recognition.py` to recognize digits from images using the trained model.
3. Run `data_entry_automation.py` to automate data entry using the trained model.

## Requirements

- Python 3.10
- TensorFlow
- Keras
- pandas
- numpy
- matplotlib
- scikit-learn
- loguru
- halo

## Installation

1. Clone the repository.
2. Install the requirements using pip: `pip install -r requirements.txt`
3. Run the scripts as described in the Usage section.

## License

This project is licensed under the MIT License.