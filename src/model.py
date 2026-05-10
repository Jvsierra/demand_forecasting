import pandas as pd

import numpy as np

from prophet import Prophet

from typing import List

from sklearn.metrics import root_mean_squared_log_error

import mlflow
from mlflow.prophet import log_model

import re

import os


def train_prophet_model(df_sales: pd.DataFrame, feature_cols: List[str] = None,
                        skew_threshold: float = None, verbose: bool = False) -> None:
    """
    Train a Prophet model for a given store and product family combination.

    Args:
        df_sales (pd.DataFrame): DataFrame containing the sales data for a specific store and product family.
        Must contain 'date' and 'sales' columns.
        feature_cols (List[str]): List of additional feature columns to include as regressors in the model. Default is None.
        skew_threshold (float): Threshold for skewness to apply log transformation to the target variable. Default is None (no transformation).
        verbose (bool): Flag to indicate whether to print additional information during training. Default is False
    """
    mlflow.set_experiment("prophet_demand_forecasting_model_training")

    with mlflow.start_run():
        store_nbr = df_sales['store_nbr'].iloc[0]
        family = df_sales['family'].iloc[0]

        # Remove special characters from family name for better model naming
        family = re.sub(r'[^A-Z]', '', family)

        df_sales = df_sales.sort_values(by='date')

        df_sales = df_sales.rename(columns={'date': 'ds', 'sales': 'y'})

        log_transformed = False

        # Apply log transformation if needed
        if skew_threshold and df_sales['y'].skew() > skew_threshold:
            df_sales['y'] = np.log1p(df_sales['y'])

            log_transformed = True

        # Train model
        m = Prophet()

        if feature_cols:
            for feature in feature_cols:
                m.add_regressor(feature)

            df_sales = df_sales[['ds', 'y'] + feature_cols]

        m.fit(df_sales)

        # Calculate training set metric

        forecast_train = m.predict(df_sales)

        fitted_values = forecast_train['yhat'].values
        fitted_values = np.clip(fitted_values, a_min=0, a_max=None)

        if log_transformed:
            y_true = np.expm1(df_sales['y'].values)
            fitted_values = np.expm1(fitted_values)
        else:
            y_true = df_sales['y'].values

        fitted_values = np.clip(fitted_values, a_min=0, a_max=None)

        train_rmsle = root_mean_squared_log_error(
            y_true,
            fitted_values
        )


        # Log model/metrics/params to MLflow
        log_model(
            pr_model=m,
            name="prophet_model",
            registered_model_name=f"demand_forecasting_prophet_model_store_{store_nbr}_family_{family}"
        )

        mlflow.log_metric('train_rmsle', train_rmsle)

        mlflow.log_params({
            'store_nbr': store_nbr,
            'family': family,
            'skew_threshold': skew_threshold,
            'verbose': verbose,
            'features': feature_cols
        })

        if verbose:
            print(f"Trained Prophet model for store {store_nbr} and family {family} with train RMSLE: {train_rmsle:.4f}")

import pandas as pd
import numpy as np

from typing import List

import mlflow


def prophet_model_inference(df_sales: pd.DataFrame, feature_cols: List[str] = None,
                            model_version: str = "latest", verbose: bool = False,
                            save_results: bool = True) -> pd.DataFrame:
    """
    Run inference using a registered Prophet model.

    Args:
        df_sales (pd.DataFrame): Sales dataframe containing:
            - date column
            - optional regressor column

        feature_cols (List[str]): Regressor columns used during training.

        model_version (str):Registered model version/stage. Examples:
            - "1"
            - "latest"
            - "Production"

        verbose (bool):
            Whether to print additional information.

        save_results (bool):
            Whether to save the inference results.


    Returns:
        pd.DataFrame: DataFrame with predictions.
    """

    # ==========================================
    # CREATE/SET INFERENCE EXPERIMENT
    # ==========================================
    mlflow.set_experiment(
        "prophet_demand_forecasting_model_inference"
    )

    with mlflow.start_run():

        # ==========================================
        # REGISTERED MODEL NAME
        # ==========================================
        store_nbr = df_sales['store_nbr'].iloc[0]
        family = df_sales['family'].iloc[0]

        # Remove special characters from family name for better model naming
        original_family = family
        family = re.sub(r'[^A-Z]', '', family)


        registered_model_name = (
            f"demand_forecasting_prophet_model_store_{store_nbr}_family_{family}"
        )

        # ==========================================
        # LOAD REGISTERED MODEL
        # ==========================================
        model_uri = (
            f"models:/{registered_model_name}/{model_version}"
        )

        model = mlflow.prophet.load_model(model_uri)

        # ==========================================
        # PREPARE INPUT DATA
        # ==========================================
        df_sales = df_sales.copy()

        df_sales = df_sales.sort_values(by='date')

        df_sales = df_sales.rename(
            columns={
                'date': 'ds'
            }
        )

        # ==========================================
        # VALIDATE REQUIRED COLUMNS
        # ==========================================
        required_columns = ['ds']

        if feature_cols:
            required_columns.extend(feature_cols)

        missing_columns = [
            col for col in required_columns
            if col not in df_sales.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Keep only necessary columns
        if feature_cols:
            df_sales = df_sales[
                ['ds'] + feature_cols
            ]
        else:
            df_sales = df_sales[['ds']]

        # ==========================================
        # RUN INFERENCE
        # ==========================================
        forecast = model.predict(df_sales)

        # ==========================================
        # FORMAT PREDICTIONS
        # ==========================================
        df_predictions = forecast[
            [
                'ds',
                'yhat',
                'yhat_lower',
                'yhat_upper'
            ]
        ].copy()

        df_predictions = df_predictions.rename(
            columns={
                'ds': 'date',
                'yhat': 'prediction',
                'yhat_lower': 'prediction_lower',
                'yhat_upper': 'prediction_upper'
            }
        )

        # Remove negative predictions
        df_predictions['prediction'] = np.clip(
            df_predictions['prediction'],
            a_min=0,
            a_max=None
        )

        df_predictions['prediction_lower'] = np.clip(
            df_predictions['prediction_lower'],
            a_min=0,
            a_max=None
        )

        df_predictions['prediction_upper'] = np.clip(
            df_predictions['prediction_upper'],
            a_min=0,
            a_max=None
        )

        # Save results and log file
        if save_results:
            os.makedirs("./data/model_results", exist_ok=True)

            predictions_file = (
                f"./data/model_results/predictions_store_{store_nbr}_family_{family}.csv"
            )

            df_predictions['store_nbr'] = store_nbr
            df_predictions['family'] = original_family

            df_predictions.to_csv(
                predictions_file,
                index=False
            )

            mlflow.log_artifact(predictions_file)

        # ==========================================
        # LOG PARAMETERS
        # ==========================================
        mlflow.log_params({
            'store_nbr': store_nbr,
            'family': family,
            'model_name': registered_model_name,
            'model_version': model_version,
            'features': feature_cols,
            'n_predictions': len(df_predictions)
        })


        # ==========================================
        # OPTIONAL LOGGING
        # ==========================================
        if verbose:
            print("=" * 60)
            print("Inference completed successfully")
            print(f"Model: {registered_model_name}")
            print(f"Version: {model_version}")
            print(f"Predictions generated: {len(df_predictions)}")
            print("=" * 60)

        return df_predictions