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
df_triad_count_swap_path = config.get('motifs representation', 'df_triad_count_swap_path')

G_dataset = get_dataset_living(dataset_path)
dataset_len = len(G_dataset)

motifs_dict = load_motifs()
available_graph_names = []

rows_tc_G_swap = []
for graph_index, G in enumerate(G_dataset):
    graph_name = G['name']

    for i in range(50):
        rows_tc_G_swap.append(process_row(i, G))
    progression_bar(graph_index +1, dataset_len)

print()
df_swap_tc = pd.DataFrame(rows_tc_G_swap, columns = ['Random network ID', 'Network'] + [motif for motif in motifs_dict])
save_df_to_pickle(df_swap_tc, df_triad_count_swap_path)
