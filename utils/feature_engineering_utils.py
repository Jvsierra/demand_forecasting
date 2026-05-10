import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils.data_loader_utils import load_sales_data

from typing import Tuple, List

FeatureTransformationResult = Tuple[pd.DataFrame, TransformerMixin]
FeatureEngineeringResult = Tuple[pd.DataFrame, List[str]]

def encode_categorical_features(df: pd.DataFrame, categorical_features: str | List[str], 
                                categorical_encoder: str = 'target_encoder',
                                add_encoded_features: bool = False,
                                target_column: str = None) -> FeatureTransformationResult:
    """
    Encode categorical features using a given
    encoding method (target encoding, one hot encoding
    or label encoding).

    Args:
        df (pd.DataFrame): input pandas DataFrame.
        categorical_features (str or List[str]): name or 
            list with the names of the categorical features.
        categorical_encoder (str, optional): method to use to
            encode the features. Can be either
            'target_encoder', 'one_hot_encoder',
            or 'label_encoder'. Defaults to
            'target_encoder'.
        add_encoded_features(bool, optional): whether to
            add the encoded features in the input
            DataFrame.
        target_column (str, optional): name of the target
            variable in the DataFrame. Only useful when
            the categorical_encoder = 'target_encoder'.
            Defaults to None.

    Raises:
        ValueError: when `categorical_encoder` is
            not one of the three mentioned.

    Returns:
        CategoricalEncodingResult: tuple with the
            transformed dataset and the fitted
            transformer.
    """
    y = None

    if categorical_encoder == 'target_encoder':
        encoder = TargetEncoder()

        y = df[target_column]
    elif categorical_encoder == 'one_hot_encoder':
        encoder = OneHotEncoder()
    elif categorical_encoder == 'label_encoder':
        encoder = LabelEncoder()
    else:
        raise ValueError(f"Encoding method {categorical_encoder} not supported")
    
    if y:
        encoder.fit(df[categorical_features], y)
    else:
        encoder.fit(df[categorical_features])

    df = df.copy()

    if add_encoded_features:
        df[encoder.get_feature_names_out()] = encoder.transform(df[categorical_features])

    return df, encoder

def scale_numerical_features(df: pd.DataFrame, numerical_features: List[str], 
                             numerical_scaler: str = 'min_max_scaler',
                             add_scaled_features: bool = False) -> FeatureTransformationResult:
    """
    Scale numerical features using a given
    scaling method (MinMaxScaler, StandardScaler).

    Args:
        df (pd.DataFrame): input pandas DataFrame.
        numerical_features (List[str]): list with
            the names of the numerical features.
        numerical_scaler (str, optional): method to use to
            scaler the features. Can be either 'min_max_scaler'
            or 'standard_scaler'.
        add_scaled_features(bool, optional): whether to
            add the scaled features in the input
            DataFrame.

    Raises:
        ValueError: when `numerical_scaler` is
            not one of the mentioned.

    Returns:
        CategoricalEncodingResult: tuple with the
            transformed dataset and the fitted
            scaler.
    """
    if numerical_scaler == 'min_max_scaler':
        scaler = MinMaxScaler()
    elif numerical_scaler == 'standard_scaler':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Scaling method {numerical_scaler} not supported")
    
    df = df.copy()
    
    scaler.fit(df[numerical_features])

    if add_scaled_features:
        df[[f"{feature_name}_scaled" for feature_name in numerical_features]] = scaler.transform(df[numerical_features])

    return df, scaler

def fill_time_series_gaps(df: pd.DataFrame, date_column: str, freq: str = 'D') -> pd.DataFrame:
    """
    Fills time series gaps by resampling the
    dataset to a given frequency.

    Args:
        df (pd.DataFrame): input pandas DataFrame.
        date_column (str): name of the date column.
        freq (str, optional): frequency to resample
            the dataset. Defaults to 'D'.

    Returns:
        pd.DataFrame: resampled DataFrame.
    """
    df = df.copy()

    df[date_column] = pd.to_datetime(df[date_column])

    df = df.set_index(date_column).sort_index()

    df = df.resample(freq).asfreq()

    # Fill null values
    categorical_cols = ['store_nbr', 'family', 'store_city', 'store_state',\
        'store_type', 'store_cluster']
    
    if 'demand_type' in df.columns:
        categorical_cols.append('demand_type')

    df[categorical_cols] = df[categorical_cols].ffill()

    event_cols = ['event_description_national', 'event_description_regional', 'event_description_local']

    df[event_cols] = df[event_cols].fillna('N/A')

    df['onpromotion'] = df['onpromotion'].fillna(0)
    df['dcoilwtico'] = df['dcoilwtico'].ffill()
    df['sales'] = df['sales'].fillna(0)


    return df.reset_index()                         


def create_features(df_sales: pd.DataFrame, scale_log: bool = False,
                    drop_columns: bool = False) -> FeatureEngineeringResult:
    """
    Creates sales training data's features.

    Args:
        df_sales (pd.DataFrame): input sales data.
        lag_amount (int, optional): amount of
            lag features to add. Defaults to False.
        scale_log (bool, optional): whether to scale
            sales' target column using log function.
            Defaults to False.
        drop_columns (bool, optional): whether to drop
            original columns. Defaults to False.
        transform_features (bool, optional): whether to
            transform the features. Defaults to False.
        
    Raise:
        ValueError: if `numerical_scaler` or `categorical_encoder`
            value is not supported.       

    Returns:
        FeatureEngineeringResult: tuple with the input
            dataset (transformed if so was informed)
            and a list with the names of the features.
    """
    feature_names = ['onpromotion', 'month',\
                     'dcoilwtico', 'event_description_national',\
                     'event_description_regional', 'event_description_local']
    
    # Fill gaps
    df_sales = pd.concat(
        [fill_time_series_gaps(df, date_column='date') for _, df in df_sales.groupby(['store_nbr', 'family'], group_keys=False)],
        ignore_index=True
    )

    # Assure `date` column is of datetime type
    df_sales['date'] = pd.to_datetime(df_sales['date'])

    # Apply log, if this option was selected
    if scale_log:
        df_sales['sales_log'] = np.log1p(df_sales['sales'])

    # Create time features
    df_sales['month'] = df_sales['date'].dt.month

    return df_sales, feature_names

    

def main():
    df_sales = load_sales_data(
        sales_data_path='./data/train.csv',
        oil_data_path='./data/oil.csv',
        holidays_events_data_path='./data/holidays_events.csv',
        stores_data_path='./data/stores.csv',
        add_demand_classification=True)
    
    df_sales_fe, _ = create_features(df_sales)

    print(df_sales_fe[df_sales_fe['date'] == '2015-12-25'])

if __name__ == '__main__':
    main()