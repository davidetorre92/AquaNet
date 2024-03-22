
import numpy as np
import pandas as pd

import os
import argparse
from configparser import ConfigParser
from ..utils.file_handlers import get_dataset_living, index_from_dataset, save_df_to_pickle

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)

node_periphery_gen_vul_df_path = config.get('core and periphery', 'gen_vul_df_path')
max_tl_structure_df_path = config.get('core and periphery', 'max_tl_structure_df_path')

# Measure
node_periphery_gen_vul_df = pd.read_pickle(node_periphery_gen_vul_df_path)
max_tl_structure_df = node_periphery_gen_vul_df[['Network', 'Trophic Level', 'Structure']].groupby(['Network', 'Structure']).max()
max_tl_structure_df.columns = ['Max Trophic Level']
# Save
save_df_to_pickle(max_tl_structure_df, max_tl_structure_df_path)