import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_na_values(data: pd.DataFrame, CONF):
    """
    Process missing values in the data.
    Args
    ----
    data : pd.DataFrame
        Data to process.
    CONF : DotMap
        Configuration object.

    Returns
    -------
    data : pd.DataFrame
        Data with missing values processed.
    """
    if CONF.data.na_values == "drop_rows":
        data = data.dropna(axis=0, how="any")
    elif CONF.data.na_values == "drop_columns":
        data = data.dropna(axis=1, how="any")
    elif CONF.data.na_values == "fillna":
        data = data.fillna(
            0
        )  # TODO: Implement a more sophisticated way to fill missing values
    assert not data.isnull().values.any()
    return data

# Define a function to split the data
def split_data(df: pd.DataFrame, column_name: str):
    """
    Split the data into train, validation, and test sets based on the year in the specified column.
    Args
    ----
    df : pd.DataFrame
        Data to split.
    column_name : str
        Name of the column containing the date information.
    Returns
    -------
    df : pd.DataFrame
        Data with additional columns indicating the split.
    """
    # Create binary columns for train, val, and test
    df["train"] = df[column_name].dt.year.isin([2019, 2020])
    df["val"] = df[column_name].dt.year == 2021
    df["test"] = df[column_name].dt.year > 2021

    return df

def normalize_data(df, ignore_features=list[str]):
    """
    Scale the data using StandardScaler.
    Args
    ----
    df : pd.DataFrame
        DataFrame to scale.
    ignore_features : list
        List of features to ignore during scaling.
    """
    def apply_scaling(df, feature):
        scaler = StandardScaler()
        train_df = df[df['train']]
        train_feature = train_df[[feature]]
        scaler.fit(train_feature)
        scaled_features = scaler.transform(df[[feature]])
        df[feature] = scaled_features
        return df

    # Apply scaling to all datasets
    feature_columns = df.columns.tolist()
    for feature in feature_columns:
        if feature not in (ignore_features + ['index'] + ['train', 'val', 'test']):
            df = apply_scaling(df, feature)
    return df

def aggregate_weather_data(df, column, copy=True):
    """
    Aggregate weather data by the specified column.
    Args
    ----
    df : pd.DataFrame
        DataFrame containing weather data.
    column : str
        Column to group by.
    copy : bool
        Whether to copy the DataFrame before processing.
    Returns
    -------
    df : pd.DataFrame
        DataFrame with weather data aggregated by the specified column.
    """
    if copy:
        df = df.copy()
    aggregations = {
        "cdir": ["min", "max", "mean"],
        "z": ["min", "max", "mean"],
        "msl": ["min", "max", "mean"],
        "blh": ["min", "max", "mean"],
        "tcc": ["min", "max", "mean"],
        "u10": ["min", "max", "mean"],
        "v10": ["min", "max", "mean"],
        "t2m": ["min", "max", "mean"],
        "ssr": ["min", "max", "mean"],
        "tsr": ["min", "max", "mean"],
        "sund": ["min", "max", "mean"],
        "tp": ["min", "max", "mean"],
        "fsr": ["min", "max", "mean"],
        "u100": ["min", "max", "mean"],
        "v100": ["min", "max", "mean"],
    }
    df = df.groupby(column).agg(aggregations)
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    return df