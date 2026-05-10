import numpy as np

import pandas as pd

def classify_demand_time_series(df_sales: pd.DataFrame) -> str:
    """
    Classify a time series in 'smooth', 'erratic',
    'lumpy' or 'intermittent' based on its
    properties.

    Args:
        df_sales (pd.DataFrame): input pandas DataFrame
            with the sales data.

    Returns:
        demand_type(str): time series' classification
    """
    # Assure `date` is on the correct type
    df_sales['date'] = pd.to_datetime(df_sales['date'])

    df_sales = df_sales.sort_values(by='date').reset_index()

    # Use only data after first sell
    first_sale_index = df_sales[df_sales['sales'] > 0].sort_index()
    
    if len(first_sale_index) > 0:
        first_sale_index = first_sale_index.index[0]
        df_sales = df_sales.iloc[first_sale_index:]

    series = df_sales['sales']

    non_zero_series = series[series > 0]

    # Intermittency classification

    if len(non_zero_series) > 1:
        cv2 = (np.std(non_zero_series) / np.mean(non_zero_series)) ** 2
    else:
        cv2 = 0

    adi = len(series) / len(non_zero_series) if len(non_zero_series) > 0 else np.inf

    if adi <= 1.32 and cv2 <= 0.49:
        demand_type = 'smooth'
    elif adi > 1.32 and cv2 <= 0.49:
        demand_type = 'intermittent'
    elif adi <= 1.32 and cv2 > 0.49:
        demand_type = 'erratic'
    else:
        demand_type = 'lumpy'

    return demand_type


def load_oil_data_and_merge(df_sales: pd.DataFrame, oil_data_path: str) -> pd.DataFrame:
    """
    Load oil data and merge it with the sales data.

    Args:
        df_sales (pd.DataFrame): pandas DataFrame with
            the sales data.
        oil_data_path (str): path to the oil data.

    Returns:
        df_sales (pd.DataFrame): original sales DataFrame
            with the oil data merged to it.
    """
    # Load oil data
    df_oil = pd.read_csv(oil_data_path)

    # Cast `date` column to datetime
    df_oil['date'] = pd.to_datetime(df_oil['date'])

    # Fill date gaps
    df_oil = df_oil.set_index('date').sort_index()

    full_idx = pd.date_range(start=df_oil.index.min(), end=df_oil.index.max(), freq='D')
    df_oil = df_oil.reindex(full_idx)

    df_oil['dcoilwtico'] = (
        df_oil['dcoilwtico']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
    )

    df_oil.index.name = 'date'
    
    df_oil = df_oil.reset_index()

    # Merge dataframes and return
    df_sales = df_sales.merge(df_oil, on='date', how='left')

    return df_sales

def load_stores_data_and_merge(df_sales: pd.DataFrame, stores_data_path: str) -> pd.DataFrame:
    """
    Load stores data and merge it to sales data.

    Args:
        df_sales (pd.DataFrame): original sales pandas DataFrame.
        stores_data_path(str): path to the stores data.

    Returns:
        df_sales (pd.DataFrame): original sales pandas DataFrame
            with the stores data added to it.
    """
    # Load data and rename coluns
    df_stores = pd.read_csv(stores_data_path)

    df_stores = df_stores.rename(columns={
        'city': 'store_city',
        'state': 'store_state',
        'type': 'store_type',
        'cluster': 'store_cluster'
    })

    # Merge and return
    df_sales = df_sales.merge(df_stores, on='store_nbr', how='left')

    return df_sales

def get_holidays_events_data_and_merge(df_sales: pd.DataFrame,
                                       holidays_events_data_path: str) -> pd.DataFrame:
    """
    Load holidays/events data and merge to sales DataFrame. 

    Args:
        df_sales (pd.DataFrame): pandas DataFrame with the sales data.
        holidays_events_data_path (str): path to the events/holidays data.

    Returns:
        df_sales (pd.DataFrame): original sales pandas DataFrame with
            the holidays/events data added to it.
    """
    # Load data
    df_holidays_events = pd.read_csv(holidays_events_data_path)

    # Cast `date_column` to datetime
    df_holidays_events['date'] = pd.to_datetime(df_holidays_events['date'])

    # Remove `Work Day` rows, since they are not
    # actually holidays or events
    df_holidays_events = df_holidays_events[df_holidays_events['type'] != 'Work Day']

    # Remove transferred rows
    df_holidays_events = df_holidays_events[df_holidays_events['transferred'] == False]

    # Rename `description` column
    # to make it more explanatory
    df_holidays_events = df_holidays_events.rename(columns={
        'description': 'event_description'
    })

    # Add national holidays - join only by date
    df_holidays_events_national = df_holidays_events[df_holidays_events['locale'] == 'National'].copy()

    # Select only needed columns
    df_holidays_events_national = df_holidays_events_national[['date', 'event_description']]

    # Group events on the same data
    df_holidays_events_national = df_holidays_events_national.groupby('date')['event_description'].apply(' | '.join).reset_index()

    df_holidays_events_national = df_holidays_events_national.rename(columns={
        'event_description': 'event_description_national'
    })

    df_sales = df_sales.merge(df_holidays_events_national, on='date', how='left')

    # Add regional holidays - join by date and state
    df_holidays_events_regional = df_holidays_events[df_holidays_events['locale'] == 'Regional'].copy()

    df_holidays_events_regional = df_holidays_events_regional[['date', 'locale_name', 'event_description']]

    df_holidays_events_regional = df_holidays_events_regional.groupby(['date', 'locale_name'])['event_description'].apply(' | '.join).reset_index()

    df_holidays_events_regional = df_holidays_events_regional.rename(columns={
        'locale_name': 'store_state',
        'event_description': 'event_description_regional'
    })

    df_sales = df_sales.merge(
        df_holidays_events_regional,
        on=['date', 'store_state'],
        how='left'
    )

    # Add city holidays/events - join by date and city
    df_holidays_events_local = df_holidays_events[df_holidays_events['locale'] == 'Local'].copy()

    df_holidays_events_local = df_holidays_events_local[['date', 'locale_name', 'event_description']]

    df_holidays_events_local = df_holidays_events_local.groupby(['date', 'locale_name'])['event_description'].apply(' | '.join).reset_index()

    df_holidays_events_local = df_holidays_events_local.rename(columns={
        'locale_name': 'store_city',
        'event_description': 'event_description_local'
    })

    df_sales = df_sales.merge(
        df_holidays_events_local,
        on=['date', 'store_city'],
        how='left'
    )

    # Fill rows with no events
    df_sales['event_description_national'] = df_sales['event_description_national'].fillna('N/A')
    df_sales['event_description_regional'] = df_sales['event_description_regional'].fillna('N/A')
    df_sales['event_description_local'] = df_sales['event_description_local'].fillna('N/A')

    return df_sales

def load_sales_data(sales_data_path: str,
                    oil_data_path: str = '../data/oil.csv',
                    stores_data_path = '../data/stores.csv',
                    holidays_events_data_path: str = '../data/holidays_events.csv',
                    min_date: str = None,
                    max_date: str = None,
                    add_demand_classification: str = False) -> pd.DataFrame:
    """
    Load sales data, add all the
    needed data and return it in a pandas
    DataFrame.

    Args:
        sales_data_path (str): path to the sales data.
        oil_data_path (str, optional): path the oil
            data. Defaults to '../data/oil.csv'.
        stores_data_path (str, optional): path to the
            stores' info data. Defaults to '../data/stores.csv'.
        holidays_events_data_path (str, optional): path to the
            holidays/events' data. Defaults to '../data/holidays_events.csv'.
        min_date (str, optional): first date to be
            considered in the data. Defaults to None.
        max_date (str, optional): last date to be
            considered in the data. Defaults to None.
        add_demand_classification (str, optional): whether to add
            the time series classification. Defaults to False.

    Returns:
        df_sales (pd.DataFrame): sales data with all the needed
            info in a pandas DataFrame object.

    """
    df_sales = pd.read_csv(sales_data_path, index_col='id')

    df_sales['date'] = pd.to_datetime(df_sales['date'])

    # Add oil price data
    df_sales = load_oil_data_and_merge(
        df_sales=df_sales,
        oil_data_path=oil_data_path
    )

    # Add stores info data
    df_sales = load_stores_data_and_merge(
        df_sales=df_sales,
        stores_data_path=stores_data_path
    )

    # Add events/holiday data
    df_sales = get_holidays_events_data_and_merge(
        df_sales=df_sales, 
        holidays_events_data_path=holidays_events_data_path
    )

    if min_date:
        df_sales = df_sales[df_sales['date'] >= min_date]

    if max_date:
        df_sales = df_sales[df_sales['date'] <= max_date]

    if add_demand_classification:
        demand_types = (
             df_sales.groupby(['store_nbr', 'family'], group_keys=False)
            .apply(classify_demand_time_series, include_groups=False)
            .reset_index()
            .rename(columns={0: 'demand_type'})
        )

        df_sales = df_sales.merge(demand_types, on=['store_nbr', 'family'], how='left')

    return df_sales

def main():
    # WARNING: run it from root folder
    df_sales = load_sales_data(
        sales_data_path='data/train.csv',
        oil_data_path='data/oil.csv',
        stores_data_path='data/stores.csv',
        holidays_events_data_path='data/holidays_events.csv',
        add_demand_classification=True
    )

    print(df_sales)

if __name__ == '__main__':
    main()