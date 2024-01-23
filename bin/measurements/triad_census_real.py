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
df_triad_count_path = config.get('motifs representation', 'df_triad_count_path')

G_dataset = get_dataset_living(dataset_path)
dataset_len = len(G_dataset)

motifs_dict = load_motifs()
available_graph_names = []

rows = []
for graph_index, G in enumerate(G_dataset):
    graph_name = G['name']

    row = []
    row.append(graph_name)
    available_graph_names.append(graph_name)
    for _, item in motif_contained_in_G_fast(G).items():
        row.append(item)

    rows.append(tuple(row))
    progression_bar(graph_index +1, dataset_len)

print()
df_triad_count = pd.DataFrame(rows, columns = ['graph_name'] + [key for key in motifs_dict])
save_df_to_pickle(df_triad_count, df_triad_count_path)
