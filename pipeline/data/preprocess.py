import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1):
        self.constant = constant

    def fit(self, X, y=None):
        # Custom logic for fitting, if necessary
        return self

    def transform(self, X):
        return X / self.constant

    def inverse_transform(self, X):
        return X * self.constant


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
    Split the data into train, validation, and test sets based
    on the year in the specified column.
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


def normalize_data(df, ignore_features=list[str], constant=None):
    """
    Scale the data using StandardScaler or custom constants.

    Args:
    ----
    df : pd.DataFrame
        DataFrame to scale.
    ignore_features : list
        List of features to ignore during scaling.
    constant : float, optional
        customized normalizing constants for each feature.

    Returns:
    -------
    df : pd.DataFrame
        Scaled DataFrame.
    scalers : dict
        Dictionary containing the scalers used for each feature.
    """

    def apply_scaling(df, feature):
        if constant:
            scaler = CustomScaler(constant=constant)
        else:
            scaler = StandardScaler()
        train_df = df[df["train"]]
        train_feature = train_df[[feature]]
        scaler.fit(train_feature)
        scaled_features = scaler.transform(df[[feature]])
        df[feature] = scaled_features
        return df, scaler

    # Apply scaling to all datasets
    flag_colums = ["train", "val", "test", "index"]
    scalers = dict()
    feature_columns = df.columns.tolist()
    for feature in feature_columns:
        if feature not in (ignore_features + flag_colums):
            df, scaler = apply_scaling(df, feature)
            scalers[feature] = scaler
    return df, scalers


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
