from numpy import save

from src.model import train_prophet_model, prophet_model_inference
from utils.data_loader_utils import load_sales_data

from tqdm import tqdm

import argparse

def run_model(min_date: str='2017-04-15', train_model: bool = True,
              save_inference_results: bool = True) -> None:
    """
    Main function to run the entire pipeline for sales forecasting using Prophet.

        Steps:
        1. Data Collection: Load the sales data, oil data, store data, and holidays/events data for both training and testing sets.
        2. Model Training: Train the Prophet model for each store and product family combination using the training data.
        3. Model Inference: Use the trained models to make predictions on the test set and evaluate the performance.

    Args:
        min_date (str): Minimum date for filtering the sales data. Default is '2017-04-15'.
        train_model (bool): Flag to indicate whether to train the model. Default is True.
        save_inference_results (bool): Flag to indicate whether to save the inference results. Default is True.
    """
    # 1. Data Collection

    # Train set
    df_train = load_sales_data(
        sales_data_path='./data/train.csv',
        oil_data_path='./data/oil.csv',
        stores_data_path='./data/stores.csv',
        holidays_events_data_path='./data/holidays_events.csv',
        min_date=min_date,
    )

    df_test = load_sales_data(
        sales_data_path='./data/test.csv',
        oil_data_path='./data/oil.csv',
        stores_data_path='./data/stores.csv',
        holidays_events_data_path='./data/holidays_events.csv',
    )

    # 2. Model Training
    if train_model:
        for group, df in tqdm(df_train.groupby(['store_nbr', 'family']), desc='Training Prophet Models'):
            try:
                    train_prophet_model(df, feature_cols=['onpromotion'])
            except Exception as ex:
                print(f"Error for granularity {group}: {ex}")

    # 3. Model Inference
    for group, df in tqdm(df_test.groupby(['store_nbr', 'family']), desc='Running Inference with Prophet Models'):
        try:
                _ = prophet_model_inference(df, feature_cols=['onpromotion'],
                                            save_results=save_inference_results)
        except Exception as ex:
            print(f"Error for granularity {group}: {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_date', type=str, default='2017-04-15', help='Minimum date for filtering the sales data.')
    parser.add_argument('--train_model', action='store_true', help='Flag to indicate whether to train the model.')
    parser.add_argument('--save_inference_results', action='store_true', help='Flag to indicate whether to save the inference results.')

    args = parser.parse_args()

    run_model(args.min_date, args.train_model, args.save_inference_results)