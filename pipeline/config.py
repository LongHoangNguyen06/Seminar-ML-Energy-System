# flake8: noqa
import os

import torch.nn as nn
from dotmap import DotMap


def get_config():
    CONF = DotMap()
    CONF.data.data_dir = "/graphics/scratch2/students/nguyenlo/seminar-ml"
    CONF.data.raw_data_dir = os.path.join(CONF.data.data_dir, "raw_data")
    CONF.data.raw_inspection_dir = os.path.join(
        CONF.data.data_dir, "raw_data_inspection"
    )
    CONF.data.preprocessed_data_dir = os.path.join(
        CONF.data.data_dir, "preprocessed_data"
    )
    CONF.data.preprocessed_data_inspection_dir = os.path.join(
        CONF.data.data_dir, "preprocessed_data_inspection"
    )

    os.makedirs(CONF.data.raw_data_dir, exist_ok=True)
    os.makedirs(CONF.data.raw_inspection_dir, exist_ok=True)
    os.makedirs(CONF.data.preprocessed_data_dir, exist_ok=True)
    os.makedirs(CONF.data.preprocessed_data_inspection_dir, exist_ok=True)

    CONF.data.na_values = "drop_columns"  # drop_rows, drop_columns, fillna
    CONF.data.inspect = False  # False to make faster
    CONF.data.process_raw_data = False
    CONF.data.price_normalization_constant = 10000.0  # euro/MWh
    CONF.data.loaded_raw_data = False  # Don't change this
    CONF.data.normalize_data = True
    CONF.data.plot = False  # False to make faster

    # Model configuration for 1 hour forecasting
    CONF.model.horizons = [1, 24]

    # Testing
    CONF.model.save_path = os.path.join(CONF.data.data_dir, "models")
    CONF.model.best_model_path = os.path.join(
        CONF.model.save_path, "run_1", "model.pth"
    )
    CONF.model.best_hyperparameter_path = os.path.join(
        CONF.model.save_path, "run_1", "hyperparameters.pth"
    )

    # Time series hyper parameters
    CONF.model.features = [
        "capacity_Biomass [MW]",
        "capacity_Hydro Power [MW]",
        "capacity_Wind Offshore [MW] ",
        "capacity_Wind Onshore [MW]",
        "capacity_Photovoltaic [MW]",
        "capacity_Other Renewable [MW]",
        "capacity_Nuclear Power [MW]",
        "capacity_Lignite [MW]",
        "capacity_Coal [MW]",
        "capacity_Natural Gas [MW]",
        "capacity_Pumped Storage [MW]",
        "capacity_Other Conventional [MW]",
        "demand_Total (Grid Load) [MWh]",
        "demand_Residual Load [MWh]",
        "demand_Pumped Storage [MWh]",
        "weather_cdir_min",
        "weather_cdir_max",
        "weather_cdir_mean",
        "weather_z_min",
        "weather_z_max",
        "weather_z_mean",
        "weather_msl_min",
        "weather_msl_max",
        "weather_msl_mean",
        "weather_blh_min",
        "weather_blh_max",
        "weather_blh_mean",
        "weather_tcc_min",
        "weather_tcc_max",
        "weather_tcc_mean",
        "weather_u10_min",
        "weather_u10_max",
        "weather_u10_mean",
        "weather_v10_min",
        "weather_v10_max",
        "weather_v10_mean",
        "weather_t2m_min",
        "weather_t2m_max",
        "weather_t2m_mean",
        "weather_ssr_min",
        "weather_ssr_max",
        "weather_ssr_mean",
        "weather_tsr_min",
        "weather_tsr_max",
        "weather_tsr_mean",
        "weather_sund_min",
        "weather_sund_max",
        "weather_sund_mean",
        "weather_tp_min",
        "weather_tp_max",
        "weather_tp_mean",
        "weather_fsr_min",
        "weather_fsr_max",
        "weather_fsr_mean",
        "weather_u100_min",
        "weather_u100_max",
        "weather_u100_mean",
        "weather_v100_min",
        "weather_v100_max",
        "weather_v100_mean",
    ]
    CONF.model.targets = [
        "supply_Biomass [MW]",
        "supply_Hydro Power [MW]",
        "supply_Wind Offshore [MW] ",
        "supply_Wind Onshore [MW]",
        "supply_Photovoltaic [MW]",
        "supply_Other Renewable [MW]",
        "supply_Pumped Storage [MW]",
    ]

    # Transformer's architecture's fixed hyperparameters
    CONF.model.num_targets = len(CONF.model.targets)
    CONF.model.num_features = len(CONF.model.features) + len(CONF.model.targets)
    CONF.model.forward_expansion = 2

    # Transformer's architecture's tunable hyperparameters
    CONF.model.num_layers = 1
    CONF.model.num_heads = 2
    CONF.model.dropout = 0.1
    CONF.model.lag = 24
    CONF.model.weather_future = 24

    # Training tunable hyperparameters
    CONF.train.batch_size = 512
    CONF.train.lr = 0.0001
    CONF.train.min_lr = 0.00001

    # Fixed hyper parameters
    CONF.train.epochs = 100
    CONF.train.loss = nn.MSELoss
    CONF.train.hyperparameters_iters = 100
    CONF.train.do_train = False
    CONF.test.do_test = True
    return CONF


CONF = get_config()
