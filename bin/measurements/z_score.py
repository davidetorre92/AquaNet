import numpy as np
import pandas as pd

import os
import argparse
from configparser import ConfigParser
from ..utils.file_handlers import get_dataset_living, progression_bar, save_df_to_pickle
from ..utils.motifs import *

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)

dataset_path = config.get('dataset', 'dataset_path')
df_tc_real_path = config.get('motifs representation', 'df_triad_count_real_path')
df_tc_random_path = config.get('motifs representation', 'df_triad_count_swap_path')
df_triad_representation_path = config.get('motifs representation', 'df_triad_representation_path')

df_tc_real = pd.read_pickle(df_tc_real_path)
df_tc_random = pd.read_pickle(df_tc_random_path)

available_graphs_real = df_tc_real['Network'].unique().tolist()
available_graphs_swap = df_tc_random['Network'].unique().tolist()

if available_graphs_real != available_graphs_swap:
    print('Triad count between real and random data differ. Aborting...')
    exit()
else:
    available_graphs = available_graphs_real

series = []
for graph_name in available_graphs:
    df_random_graph = df_tc_random[df_tc_random['Network'] == graph_name].iloc[:,2:]
    N_real = df_tc_real[df_tc_real['Network'] == graph_name].iloc[0,1:]
    mean_N_rand = df_random_graph.mean()
    sigma_N_rand = df_random_graph.std()
    series.append((N_real - mean_N_rand) / sigma_N_rand)  # cfr. eq (2.1) - Stouffer 2007

df_triad_representation = pd.DataFrame(series)
df_triad_representation.insert(0, 'Network', available_graphs)
df_triad_representation.fillna(0, inplace = True)

save_df_to_pickle(df_triad_representation, df_triad_representation_path)

