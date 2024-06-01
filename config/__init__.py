# flake8: noqa
import os

from dotmap import DotMap

CONF = DotMap()
CONF.data_processing = DotMap()
CONF.data_processing.data_dir = "/home/long/Desktop/ml-energy/tmp"
CONF.data_processing.raw_data_dir = os.path.join(CONF.data_processing.data_dir, "data")
CONF.data_processing.inspection_dir = os.path.join(
    CONF.data_processing.data_dir, "data_inspection"
)
CONF.data_processing.preprocessed_data_dir = os.path.join(
    CONF.data_processing.data_dir, "data_preprocessing"
)

os.makedirs(CONF.data_processing.raw_data_dir, exist_ok=True)
os.makedirs(CONF.data_processing.inspection_dir, exist_ok=True)
os.makedirs(CONF.data_processing.preprocessed_data_dir, exist_ok=True)

CONF.data_processing.na_values = "drop_rows"  # drop_rows, drop_colums, fillna
