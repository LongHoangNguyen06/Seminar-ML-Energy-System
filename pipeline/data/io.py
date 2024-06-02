import os
from datetime import datetime

import pandas as pd

CAPACITY_FORMAT = "%d.%m.%y"
STANDARD_FORMAT = "%d.%m.%y %H:%M"
WEATHER_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_COLUMNS = ["Date from", "Date to"]
DATE_COLUMNS_WEATHER = ["forecast_origin", "time"]


def load_data(CONF, data_type="raw"):
    """
    Load all data from the data folder.
    Args
    ----
    CONF : DotMap
        Configuration object.
    data_type : str
        Type of data to load. Options are:
        - raw
        - preprocessed
    Returns
    -------
    Installed_Capacity_Germany : pd.DataFrame
        Installed Installed_Capacity_Germany of power plants in Germany.
    Prices_Europe : pd.DataFrame
        Electricity Prices_Europe in Europe.
    Realised_Supply_Germany : pd.DataFrame
        Realised_Supply_Germany of electricity in Germany.
    Realised_Demand_Germany : pd.DataFrame
        Realised_Demand_Germany of electricity in Germany.
    Weather_Data_Germany : pd.DataFrame
        Weather data in Germany.
    """
    if data_type == "raw":
        ROOT_DIR = CONF.data.raw_data_dir
    elif data_type == "preprocessed":
        ROOT_DIR = CONF.data.preprocessed_data_dir
    else:
        raise ValueError("data_type must be either 'raw' or 'preprocessed'")

    path = os.path.join(ROOT_DIR, "Installed_Capacity_Germany.csv")
    Installed_Capacity_Germany = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), CAPACITY_FORMAT),
    )
    print(f"Loaded Installed_Capacity_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Prices_Europe.csv")
    Prices_Europe = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), STANDARD_FORMAT),
    )
    print(f"Loaded Prices_Europe from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Realised_Supply_Germany.csv")
    Realised_Supply_Germany = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), STANDARD_FORMAT),
    )
    print(f"Loaded Realised_Supply_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Reaslised_Demand_Germany.csv")
    Realised_Demand_Germany = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), STANDARD_FORMAT),
    )
    print(f"Loaded Realised_Demand_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Weather_Data_Germany.csv")
    Weather_Data_Germany = pd.read_csv(
        path,
        sep=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS_WEATHER,
        date_parser=lambda x: datetime.strptime(x.strip(), WEATHER_FORMAT),
    )
    print(f"Loaded Weather_Data_Germany from '{path}' successfully.")

    return (
        Installed_Capacity_Germany,
        Prices_Europe,
        Realised_Supply_Germany,
        Realised_Demand_Germany,
        Weather_Data_Germany,
    )


def save_data(
    Installed_Capacity_Germany,
    Prices_Europe,
    Realised_Supply_Germany,
    Realised_Demand_Germany,
    Weather_Data_Germany,
    CONF,
    data_type="raw",
):
    """
    Load all data from the data folder.
    Args
    ----
    Installed_Capacity_Germany : pd.DataFrame
        Installed Installed_Capacity_Germany of power plants in Germany.
    Prices_Europe : pd.DataFrame
        Electricity Prices_Europe in Europe.
    Realised_Supply_Germany : pd.DataFrame
        Realised_Supply_Germany of electricity in Germany.
    Realised_Demand_Germany : pd.DataFrame
        Realised_Demand_Germany of electricity in Germany.
    Weather_Data_Germany : pd.DataFrame
        Weather data in Germany.
    CONF : DotMap
        Configuration object.
    data_type : str
        Type of data to load. Options are:
        - preprocessed
    """
    if data_type == "preprocessed":
        ROOT_DIR = CONF.data.preprocessed_data_dir
    else:
        raise ValueError("data_type must be 'preprocessed'")

    # Define a general function to format datetime columns before saving
    def format_datetime(df, datetime_columns, format):
        for col in datetime_columns:
            if col in df.columns:
                df[col] = df[col].dt.strftime(format)

    # Format datetime columns
    format_datetime(Installed_Capacity_Germany, DATE_COLUMNS, CAPACITY_FORMAT)
    format_datetime(Prices_Europe, DATE_COLUMNS, STANDARD_FORMAT)
    format_datetime(Realised_Supply_Germany, DATE_COLUMNS, STANDARD_FORMAT)
    format_datetime(Realised_Demand_Germany, DATE_COLUMNS, STANDARD_FORMAT)
    format_datetime(Weather_Data_Germany, DATE_COLUMNS_WEATHER, WEATHER_FORMAT)

    path = os.path.join(ROOT_DIR, "Installed_Capacity_Germany.csv")
    Installed_Capacity_Germany.to_csv(
        path,
        sep=";",
        decimal=",",
        index=False,
    )

    path = os.path.join(ROOT_DIR, "Prices_Europe.csv")
    Prices_Europe.to_csv(
        path,
        sep=";",
        decimal=",",
        index=False,
    )

    path = os.path.join(ROOT_DIR, "Realised_Supply_Germany.csv")
    Realised_Supply_Germany.to_csv(
        path,
        sep=";",
        decimal=",",
        index=False,
    )

    path = os.path.join(ROOT_DIR, "Reaslised_Demand_Germany.csv")
    Realised_Demand_Germany.to_csv(
        path,
        sep=";",
        decimal=",",
        index=False,
    )

    path = os.path.join(ROOT_DIR, "Weather_Data_Germany.csv")
    Weather_Data_Germany.to_csv(
        path,
        sep=",",
        index=False,
    )
