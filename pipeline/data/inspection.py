import os

from ydata_profiling import ProfileReport


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
