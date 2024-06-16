import os

from ydata_profiling import ProfileReport
import pandas as pd

def save_data_inspection(
    Installed_Capacity_Germany,
    Prices_Europe,
    Realised_Supply_Germany,
    Realised_Demand_Germany,
    Weather_Data_Germany,
    CONF,
    data_type="raw",
    Weather_Data_Germany_2022=None,
):
    """
    Save data inspection reports for all data.
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
    Weather_Data_Germany_2022 : pd.DataFrame
        Weather data in Germany for 2022.
    CONF : DotMap
        Configuration object.
    data_type : str
        Type of data to load. Options are:
        - raw
        - preprocessed
    """

    if data_type == "raw":
        ROOT_DIR = CONF.data.raw_inspection_dir
    elif data_type == "preprocessed":
        ROOT_DIR = CONF.data.preprocessed_data_inspection_dir
    else:
        raise ValueError("data_type must be either 'raw' or 'preprocessed'")

    ProfileReport(
        Installed_Capacity_Germany,
        title=f"{data_type} Installed_Capacity_Germany",
        progress_bar=False,
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Installed_Capacity_Germany.html",
        )
    )

    ProfileReport(
        Prices_Europe, title=f"{data_type} Prices_Europe", progress_bar=False
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Prices_Europe.html",
        )
    )

    ProfileReport(
        Realised_Supply_Germany,
        title=f"{data_type} Realised_Supply_Germany",
        progress_bar=False,
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Realised_Supply_Germany.html",
        )
    )

    ProfileReport(
        Realised_Demand_Germany,
        title=f"{data_type} Realised_Demand_Germany",
        progress_bar=False,
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Realised_Demand_Germany.html",
        )
    )

    ProfileReport(
        Weather_Data_Germany,
        title=f"{data_type} Weather_Data_Germany",
        progress_bar=False,
        minimal=True,  # noqa: E501
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Weather_Data_Germany.html",
        )
    )

    if Weather_Data_Germany_2022 is not None:
        ProfileReport(
            Weather_Data_Germany_2022,
            title=f"{data_type} Weather_Data_Germany_2022",
            progress_bar=False,
            minimal=True,  # noqa: E501
        ).to_file(
            os.path.join(
                ROOT_DIR,
                "Weather_Data_Germany_2022.html",
            )
        )
    print(f"Saved {data_type} data inspection successfully.")

def date_range_and_resolution(df: pd.DataFrame, date_columns: list[str]):
    """
    Calculate the date range and resolution of the data.
    Args
    ----
    df : pd.DataFrame
        DataFrame to inspect.
    date_columns : list
        List of date columns in the DataFrame.
    """
    for date_column in date_columns:
        # Convert the column to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column])

        # Calculate minimum and maximum date
        min_date = df[date_column].min()
        max_date = df[date_column].max()

        # Calculate differences between consecutive dates and find the most common
        df_sorted = df.sort_values(by=date_column)
        df_sorted["date_diff"] = (
            df_sorted[date_column].diff().dt.total_seconds()
        )  # this is in seconds
        resolution = df_sorted["date_diff"].mode()[
            0
        ]  # The most common difference in seconds
        print(f"Min {date_column}: {min_date}")
        print(f"Max {date_column}: {max_date}")
        print(f"Resolution {date_column}: {pd.to_timedelta(resolution, unit='s')}")
