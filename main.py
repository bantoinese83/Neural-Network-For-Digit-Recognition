# main.py

from config import TRAIN_CSV_PATH, TEST_CSV_PATH, CLEANED_CSV_PATH, MODEL_PATH
from data_load import DataLoader
from data_visualization import visualize_data
from evaluate import evaluate_model
from train import train_model
from utils import save_data, save_model
from loguru import logger
from halo import Halo


def main():
    spinner = Halo(text='Loading', spinner='dots')

    # Create DataLoader instances
    spinner.start('Loading data...')
    data_loader_train = DataLoader(TRAIN_CSV_PATH)
    data_loader_test = DataLoader(TEST_CSV_PATH)
    spinner.succeed('Data loaded.')

    # Get data from DataLoader instances
    df_train = data_loader_train.df
    df_test = data_loader_test.df

    # Display data for a training set
    logger.info("Training Data:")
    data_loader_train.display_data()
    data_loader_train.display_column_info()
    data_loader_train.display_missing_values()
    data_loader_train.display_unique_values()
    data_loader_train.display_statistics()

    # Clean training data
    spinner.start('Cleaning training data...')
    data_loader_train.remove_duplicates()
    data_loader_train.remove_outliers()
    data_loader_train.fill_missing_values()
    spinner.succeed('Training data cleaned.')

    # Save the cleaned training data
    spinner.start('Saving cleaned training data...')
    save_data(df_train, CLEANED_CSV_PATH)
    spinner.succeed('Cleaned training data saved.')

    # Load the cleaned training data
    spinner.start('Loading cleaned training data...')
    data_loader_train_cleaned = DataLoader(CLEANED_CSV_PATH)
    df_train_cleaned = data_loader_train_cleaned.df
    spinner.succeed('Cleaned training data loaded.')

    # Visualize data
    spinner.start('Visualizing data...')
    visualize_data(df_train_cleaned)
    spinner.succeed('Data visualized.')

    # Train model
    spinner.start('Training model...')
    model, history = train_model(df_train_cleaned)
    spinner.succeed('Model trained.')

    # Save model
    spinner.start('Saving model...')
    save_model(model, MODEL_PATH)
    spinner.succeed('Model saved.')

    # Only proceed to evaluation if the model is trained successfully
    if model is not None and history is not None:
        # Print the shape of df_test before evaluation
        logger.info(f"Shape of df_test before evaluation: {df_test.shape}")

        # Evaluate model
        spinner.start('Evaluating model...')
        evaluate_model(df_test, model)
        spinner.succeed('Model evaluated.')


if __name__ == '__main__':
    main()
