import os

import pandas as pd


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

    Installed_Capacity_Germany = pd.read_csv(
        f"{ROOT_DIR}/Installed_Capacity_Germany.csv",
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    print("Loaded Installed_Capacity_Germany successfully.")

    Prices_Europe = pd.read_csv(
        f"{ROOT_DIR}/Prices_Europe.csv",
        sep=";",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    print("Loaded Prices_Europe successfully.")

    Realised_Supply_Germany = pd.read_csv(
        f"{ROOT_DIR}/Realised_Supply_Germany.csv",
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    print("Loaded Realised_Supply_Germany successfully.")

    Realised_Demand_Germany = pd.read_csv(
        f"{ROOT_DIR}/Realised_Demand_Germany.csv",
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    print("Loaded Realised_Demand_Germany successfully.")

    Weather_Data_Germany = pd.read_csv(
        f"{ROOT_DIR}/Weather_Data_Germany.csv",
        sep=",",
        na_values=["-"],
        parse_dates=["forecast_origin", "time"],
    )
    print("Loaded Weather_Data_Germany successfully.")

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

    Installed_Capacity_Germany.to_csv(
        os.path.join(ROOT_DIR, "Installed_Capacity_Germany.csv"),
        sep=";",
        decimal=",",
        index=False,
    )

    Prices_Europe.to_csv(
        os.path.join(ROOT_DIR, "Prices_Europe.csv"),
        sep=";",
        decimal=",",
        index=False,
    )

    Realised_Supply_Germany.to_csv(
        os.path.join(ROOT_DIR, "Realised_Supply_Germany.csv"),
        sep=";",
        decimal=",",
        index=False,
    )

    Realised_Demand_Germany.to_csv(
        os.path.join(ROOT_DIR, "Realised_Demand_Germany.csv"),
        sep=";",
        decimal=",",
        index=False,
    )

    Weather_Data_Germany.to_csv(
        os.path.join(ROOT_DIR, "Weather_Data_Germany.csv"),
        sep=",",
        index=False,
    )
