# flake8: noqa
import os

from dotmap import DotMap

CONF = DotMap()
CONF.data.data_dir = "/graphics/scratch2/students/nguyenlo/seminar-ml"
CONF.data.raw_data_dir = os.path.join(CONF.data.data_dir, "raw_data")
CONF.data.raw_inspection_dir = os.path.join(CONF.data.data_dir, "raw_data_inspection")
CONF.data.preprocessed_data_dir = os.path.join(CONF.data.data_dir, "preprocessed_data")
CONF.data.preprocessed_data_inspection_dir = os.path.join(
    CONF.data.data_dir, "preprocessed_data_inspection"
)

os.makedirs(CONF.data.raw_data_dir, exist_ok=True)
os.makedirs(CONF.data.raw_inspection_dir, exist_ok=True)
os.makedirs(CONF.data.preprocessed_data_dir, exist_ok=True)
os.makedirs(CONF.data.preprocessed_data_inspection_dir, exist_ok=True)

CONF.data.na_values = "drop_columns"  # drop_rows, drop_columns, fillna
CONF.data.inspect = False
CONF.data.process_raw_data = True
CONF.data.price_normalization_constant = 10000.0  # euro/MWh
CONF.data.loaded_raw_data = False
